import torch
import numpy as np
import constants as c


"""Normalize the data given the dataset. Only ImageNet and CIFAR-10 are supported"""
def transform(img, dataset='imagenet'):
    # Data
    if dataset == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406], device=c.DEVICE).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img)
        std = torch.tensor([0.229, 0.224, 0.225], device=c.DEVICE).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img)
    elif dataset == 'cifar':
        mean = torch.tensor([0.485, 0.456, 0.406], device=c.DEVICE).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img)
        std = torch.tensor([0.229, 0.224, 0.225], device=c.DEVICE).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img)
    else:
        raise "dataset is not supported"
    return ((img - mean) / std).float()


"""Given [label] and [dataset], return a random label different from [label]"""
def random_label(label, dataset='imagenet'):
    if dataset == 'imagenet':
        class_num = 1000
    elif dataset == 'cifar':
        class_num = 10
    else:
        raise "dataset is not supported"
    attack_label = np.random.randint(class_num)
    while label == attack_label:
        attack_label = np.random.randint(class_num)
    return attack_label

"""Given the variance of zero_mean Gaussian [n_radius], return a noisy version of [img]"""
def noisy_img(img, n_radius):
    return img + n_radius * torch.randn_like(img)
# Returns a tensor with the same size as input that is filled with random numbers
# from a normal distribution with mean 0 and variance 1

class Noisy(torch.autograd.Function):
    @staticmethod
    def forward(self, img, n_radius):
        return noisy_img(img, n_radius=n_radius)

    @staticmethod
    def backward(self, grad_output):
        # tohle asi vraci gradienty
        return grad_output, None
