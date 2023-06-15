import torch
import torchvision
import numpy as np
from os import listdir
from PIL import Image
import os
import pandas as pd
import constants as c

from torchvision.models import resnet101, resnet152, ResNet101_Weights, ResNet152_Weights

from utils import transform

BEST_PATH = 'samples_ADV/inception/WhiteBoxMutation/nearest_0.03_28_ssim'
CSV = 'output_tiles28_maxl1-0.03_nearest_ssim.csv'

df = pd.read_csv(os.path.join(BEST_PATH, CSV))
nums = [int(f[:f.find('_')]) for f in listdir(BEST_PATH) if f[-3:] == '.pt']


def get_labels(model_name):

    if model_name == 'R101':
        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        model = torch.nn.DataParallel(model)
    elif model_name == 'R152':
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
        model = torch.nn.DataParallel(model)
    elif model_name == 'R152adv':
        model = torchvision.models.resnet152()
        model.load_state_dict(torch.load('res152-adv.checkpoint'))
    else: raise ValueError('Unknown model.')

    model.to(c.DEVICE)
    model.eval()

    df[model_name] = np.nan
    df[model_name+'_label'] = np.nan
    df[model_name + '_orig_label'] = np.nan

    for img_num in nums:
        if model_name == 'R152adv':

            img = torch.load(os.path.join(BEST_PATH, str(img_num) + '_img.pt'), map_location=torch.device(c.DEVICE))
            img = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img).unsqueeze(0).to(dtype=torch.float)
            permute = [2, 1, 0]
            img = img[:, permute]
            label = model(img).data.max(1, keepdim=True)[1][0].item()

            img_orig = torch.load(os.path.join('samples_REAL', 'inception', str(img_num) + '_img.pt'), map_location=torch.device(c.DEVICE))[0]
            img_orig = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_orig).unsqueeze(0).to(dtype=torch.float)
            img_orig = img_orig[:, permute]
            label_orig = model(img_orig).data.max(1, keepdim=True)[1][0].item()
        
        else:
            img = torch.load(os.path.join(BEST_PATH, str(img_num) +'_img.pt'), map_location=torch.device(c.DEVICE)).unsqueeze(0)            
            label = model(transform(img.clone())).data.max(1, keepdim=True)[1][0].item()

            img_orig = torch.load(os.path.join('samples_REAL', 'inception', str(img_num) + '_img.pt'), map_location=torch.device(c.DEVICE))
            label_orig = model(transform(img_orig.clone())).data.max(1, keepdim=True)[1][0].item()

        df.at[img_num, model_name + '_label'] = label
        df.at[img_num, model_name + '_orig_label'] = label_orig
        
        true_label = torch.load(os.path.join('samples_REAL', 'inception', str(img_num) + '_label.pt'),
                                map_location=torch.device('cpu')).item()
   
        if true_label == label: df.at[img_num, model_name] = 0
        else: df.at[img_num, model_name] = 1
        del img
    
    df.to_csv(os.path.join(BEST_PATH, CSV[:-4]+'_extented.csv'), index=False)
    print(model_name, np.mean(df[model_name]))


if __name__ == "__main__":

    models = ['R101', 'R152', 'R152adv']
    for m in models:
        get_labels(m)