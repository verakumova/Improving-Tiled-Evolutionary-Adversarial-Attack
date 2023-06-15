import matplotlib.pyplot as plt
import constants as c
import torch
import os
import numpy as np
from detect import l1_detection, targeted_detection, untargeted_detection
import torch.nn.functional as func
from utils import transform


class Image:
    def __init__(self, img=None) -> None:
        self.img = img
               
    def show(self, title: str):
        view_data = torch.squeeze(self.img, 0).cpu()
        plt.imshow(view_data.permute(1, 2, 0).numpy())
        plt.title(title)
        plt.show()

    def get_img_values(self, model):
        # parameters from authors implementation
        l1_radius = 0.01 if c.NETWORK == 'vgg_small' else 0.1
        t_lr = 0.0005 if c.NETWORK == 'vgg_small' else 0.005
        ut_lr = 1 if c.NETWORK == 'vgg_small' else 0.1 if c.NETWORK == 'resnet' else 3
        steps_radius = 0.5 if c.NETWORK == 'vgg_small' else 0.03
        noise_val = []
        for _ in range(10):
            val = l1_detection(model, self.img, c.DATASET, l1_radius)
            noise_val.append(val)
        targeted_steps = targeted_detection(model, self.img, c.DATASET, t_lr, steps_radius, cap=c.T_CAP)
        untargeted_steps = untargeted_detection(model, self.img, c.DATASET, ut_lr, steps_radius, cap=c.UT_CAP)
        return min(noise_val), np.mean(noise_val), max(noise_val), targeted_steps, untargeted_steps

    def entropy(self, model):
        output = func.softmax(model(transform(self.img.clone(), dataset=c.DATASET)), dim=1)
        return - output.mul(output.log2()).sum().item()
    

class TrueImage(Image):
    def __init__(self, img_num: int, img=None) -> None:
        super().__init__(img)
        self.img_num = img_num
        if img: self.img = img
        else: self.img = torch.load(os.path.join(c.FOLDER_REAL, c.NETWORK, str(img_num) + '_img.pt'), map_location=torch.device(c.DEVICE))
        self.true_label = torch.load(os.path.join(c.FOLDER_REAL, c.NETWORK, str(img_num) + '_label.pt'), map_location=torch.device(c.DEVICE)).item()


class AdvImage(Image):
    def __init__(self, img_num: int, img) -> None:
        super().__init__(img)
        self.orig_img = TrueImage(img_num=img_num)