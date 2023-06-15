import torch
from torch.nn.modules import Upsample
import torch.nn.functional as func
from torchvision.models import resnet101, ResNet101_Weights
import random
import numpy as np
import os
from utils import transform
from deap import base, creator, tools
import constants as c
from detect import l1_detection
from image import TrueImage, AdvImage
from piq import ssim
import sys


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
# categories https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

class EvolutionaryAttack:
    def __init__(self, img:TrueImage, model, num_tiles, max_l1, mode: str, dist: str):
        assert dist.lower() in ('l2', 'ssim'), 'Distance type not allowed.'

        self.img = img
        self.model = model ## model to be fooled
        self.num_tiles = num_tiles ## number of tiles in adv image
        self.max_l1 = max_l1 ## max l1 size of noise per pixel 
        self.mode = mode ## interpolation mode for upsampling individual
        self.dist = dist

        self.aux_model = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.aux_model = torch.nn.DataParallel(self.aux_model)
        self.aux_model.to(c.DEVICE)
        self.aux_model.eval()

    def get_img_from_ind(self, ind):
        
        upsampler = Upsample(size=(c.IMG_SIZE, c.IMG_SIZE), mode=self.mode)
        noise = torch.from_numpy(np.float32(ind)).reshape([1, 3, self.num_tiles, self.num_tiles]).to(torch.device(c.DEVICE))
        noise = torch.clamp(noise, min=-self.max_l1, max=self.max_l1)
        if self.num_tiles != c.IMG_SIZE:
            adv_img = self.img.img + upsampler(noise)
        else:
            adv_img = self.img.img + noise
        adv_img = AdvImage(img_num = self.img.img_num, img = torch.clamp(adv_img, 0, 1))
        label = self.model(transform(adv_img.img.clone(), dataset=c.DATASET)).data.max(1, keepdim=True)[1][0].item()
        return adv_img, label
    
    def mutate(self, ind):
        """Mutation method which upsample first, 
        then computes gradients and them uses avg pooling for downsampling"""
        adv_img, _ = self.get_img_from_ind(ind)
        adv_img.img.requires_grad_()

        output = self.aux_model(adv_img.img)
        # output = self.model(adv_img.img)
        loss = torch.nn.CrossEntropyLoss()(output, output.argmax(dim=1))
        self.aux_model.zero_grad()
        loss.backward()
        
        gradient = adv_img.img.grad.data
        mutation = 0.01 * gradient.sign()

        downsample_factor = int(c.IMG_SIZE/self.num_tiles)
        downsample_layer = torch.nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor)
        new_individual = downsample_layer(mutation).cpu().detach().numpy().flatten()
     
        return creator.Individual(ind - new_individual), creator.Individual(ind + new_individual)

    def fitness(self, ind):
        adv_img, predicted_label = self.get_img_from_ind(ind)
        l1_radius = 0.1 #from authors implementation
        probs = func.softmax(self.model(transform(adv_img.img, dataset=c.DATASET)), dim=1)
        if predicted_label == adv_img.orig_img.true_label:
            ret = probs[0][adv_img.orig_img.true_label].item() * 10000
            return (ret,)
        else: 
            if self.dist == 'l2':       
                img_dist = torch.dist(adv_img.orig_img.img, adv_img.img).item()
            elif self.dist == 'ssim':
                # value of 1 means same images, -1 different
                img_dist = -ssim(adv_img.orig_img.img, adv_img.img, data_range=1.0).item()
            noise_val = []
            for _ in range(10):
                val = l1_detection(self.model, adv_img.img, c.DATASET, l1_radius)
                noise_val.append(val)
            ret = 100 * img_dist + 100 * max(noise_val) - 10 * probs[0][predicted_label].item()
            if ret <= 0: ret = sys.float_info.min
            return (np.log(ret),)

    def differential_evolution(self, num_gen, pop_size, cr, f, white_box_mutation=False, save=True, early_stop=False):
        ndim = 3*self.num_tiles*self.num_tiles
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.normal, 0, self.max_l1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, ndim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selRandom, k=3)
        toolbox.register("evaluate", self.fitness)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        name = '{}_tiles{}_ngen{}_popsize{}_cr{}_f{}.txt'.format(self.img.img_num, self.num_tiles, num_gen, pop_size, cr, f)

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)
        print(logbook.stream)
        last_gen = num_gen
        last_fits = []

        white_box_mutation_prob = 0.5

        for g in range(1, num_gen):
            for k, agent in enumerate(pop):
                a, b, c = toolbox.select(pop)
                y = toolbox.clone(agent)
                index = random.randrange(ndim)
                for i, value in enumerate(agent):
                    if i == index or random.random() < cr:
                        y[i] = a[i] + f * (b[i] - c[i])
                
                y.fitness.values = toolbox.evaluate(y) 
                adepts = [y, agent]
                fits = [y.fitness, agent.fitness]

                # white box mutation
                if white_box_mutation and random.random() < white_box_mutation_prob:
                    y_mutated1, y_mutated2 = self.mutate(y) 
                    agent_mutated1, agent_mutated2 = self.mutate(agent)

                    y_mutated1.fitness.values = toolbox.evaluate(y_mutated1) 
                    y_mutated2.fitness.values = toolbox.evaluate(y_mutated2) 
                    agent_mutated1.fitness.values = toolbox.evaluate(agent_mutated1) 
                    agent_mutated2.fitness.values = toolbox.evaluate(agent_mutated2) 

                    adepts.extend([y_mutated1, y_mutated2, agent_mutated1, agent_mutated2])
                    fits.extend([y_mutated1.fitness, y_mutated2.fitness, agent_mutated1.fitness, agent_mutated2.fitness])   

                max_fitness_ind = np.argmax(fits)
                pop[k] = adepts[max_fitness_ind]
            hof.update(pop)  # commented lines 533-538 in support.py, HOF
            if hof[0].fitness.values[0] < 5000: white_box_mutation_prob = 0.1
            # value 5000 is defined by fitness function
            record = stats.compile(pop)
            logbook.record(gen=g, evals=len(pop), **record)
            print(logbook.stream)
            if early_stop:
                last_fits.append(hof[0].fitness.values[0])
                if hof[0].fitness.values[0] < 5:
                    last_gen = g
                    break
                if len(last_fits) >= 15:
                    if np.mean(last_fits).round(2) == np.round(last_fits[-1], 2):
                        last_gen = g
                        break
                    else:
                        del last_fits[0]
        adv_img, predicted_label = self.get_img_from_ind(hof[0])
        if save:
            target_dir = os.path.join(c.FOLDER_ADV, c.NETWORK, 'Evolution') ## dir for saving adv images
            os.makedirs(target_dir, exist_ok=True)
            torch.save(adv_img.img, os.path.join(target_dir, name + '_label' + str(predicted_label) + '.pt'))
            output_file = open(os.path.join(target_dir, name), 'w+')
            output_file.write(logbook.__str__(0))
            output_file.close()
        print("Fitness of best individual is", hof[0].fitness.values[0])
        return adv_img, predicted_label, hof[0].fitness.values[0], last_gen