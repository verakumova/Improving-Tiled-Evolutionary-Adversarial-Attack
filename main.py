import constants as c
import torch
from csv import writer

import torch.nn.functional as func
from torchvision.utils import save_image
import pandas as pd
import os
from utils import transform
import time
import torchvision.models as models

from evo_attack import EvolutionaryAttack
from image import TrueImage, AdvImage


dtype = torch.float
device = torch.device(c.DEVICE)


if __name__ == '__main__':

    NUM_GEN = 100
    POP_SIZE = 30
    CR = 0.75
    F = 0.8
    NUM_TILES = 28
    MAX_L1 = 0.03
    DISTANCE_METRIC = 'l2'
    ATTACK_NAME = 'WhiteBoxMutation'

    model = models.inception_v3(weights='DEFAULT')
    # model = models.vgg19(weights='VGG19_Weights.DEFAULT')
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    modes = ['nearest', 'bilinear', 'bicubic']

    for mode in modes:
        dir = os.path.join(c.FOLDER_ADV, c.NETWORK, ATTACK_NAME, mode+'_'+str(MAX_L1)+'_'+str(NUM_TILES)+'_'+DISTANCE_METRIC)
        os.makedirs(dir, exist_ok=True)
        times = []
        fits = []
        min_noise_vals = []
        mean_noise_vals = []
        max_noise_vals = []
        target_steps = []
        untarget_steps = []
        entropies = []
        labels = []
        gen_num = []
        true_labels = []
        entropies_orig = []
        probabilities = []
        distances = []
        upbd = 851 if c.NETWORK == 'inception' \
            else 901 if c.NETWORK == 'resnet' \
            else 779 if c.NETWORK == 'resnet152' \
            else  809 if c.NETWORK == 'vgg' else 902

        img_nums = list(range(100))

        name = os.path.join(dir, 'output_tiles{}_maxl1-{}_{}_{}.csv'.format(NUM_TILES, MAX_L1, mode, DISTANCE_METRIC))
        dict = {'img': [], 'true_label': true_labels, 'orig_entropy': entropies_orig, 'class_prob': probabilities,
                'time': times, 'generation': gen_num, 'fitness': fits, 'label': labels,
                'min_noise': min_noise_vals, 'mean_noise': mean_noise_vals, 'max_noise': max_noise_vals,
                't_steps': target_steps, 'u_steps': untarget_steps, 'entropy': entropies, 'distance': distances}
        df = pd.DataFrame(dict)

        if os.path.exists(name):
            print(f"File '{name}' already exists.")
        else:
            df.to_csv(name, index=False)


        for i, img_num in enumerate(img_nums):
            img = TrueImage(img_num)
            true_labels.append(img.true_label)

            Attack = EvolutionaryAttack(img, model, NUM_TILES, MAX_L1, mode=mode, dist=DISTANCE_METRIC)

            entropies_orig.append(img.entropy(model))
            probs = func.softmax(model(transform(img.img, dataset=c.DATASET)), dim=1)
            probabilities.append(probs[0][img.true_label].item())

            start = time.time()
            adv_img, adv_label, best_fit, last_gen = Attack.differential_evolution(NUM_GEN, POP_SIZE, CR, F, white_box_mutation=True, save=False, early_stop=True)
            end = time.time()
            distances.append(torch.dist(img.img, adv_img.img).item())
            min_noise_val, mean_noise_val, max_noise_val, targeted_steps, untargeted_steps = adv_img.get_img_values(model)
            times.append((end-start)/60)
            fits.append(best_fit)
            min_noise_vals.append(min_noise_val)
            mean_noise_vals.append(mean_noise_val)
            max_noise_vals.append(max_noise_val)
            target_steps.append(targeted_steps)
            untarget_steps.append(untargeted_steps)
            entropies.append(adv_img.entropy(model))
            labels.append(adv_label)
            gen_num.append(last_gen)

            print('Finished {} in {:.2f} min'.format(i+1, times[-1]))
            new_row = [img_num, true_labels[-1], entropies_orig[-1], probabilities[-1], times[-1], gen_num[-1],
                    fits[-1], labels[-1],
                    min_noise_vals[-1], mean_noise_vals[-1], max_noise_vals[-1], target_steps[-1],
                    untarget_steps[-1], entropies[-1], distances[-1]]
            if true_labels[-1] == labels[-1]:
                print(img_num, 'FAILED')
            else:
                print(img_num, 'SUCCESS')
                img_to_save = adv_img.img[0]
                save_image(img_to_save, os.path.join(dir, str(img_num) + '_img.png'))
                # save_image(img, os.path.join(dir, str(img_num) + '_img.mat'))
                torch.save(img_to_save, os.path.join(dir, str(img_num) + '_img.pt'))
            with open(name, 'a+') as file:
                csv_writer = writer(file)
                csv_writer.writerow(new_row)

