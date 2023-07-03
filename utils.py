import yaml
from argparse import Namespace

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from matplotlib.figure import Figure

def load_yaml(file: str) -> Namespace:
    with open(file) as f:
        config = yaml.safe_load(f)
    return Namespace(**config)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_output_shape(model: nn.Module, input_shape: tuple):
    t = torch.rand(input_shape)
    return model(t).shape

def filename_to_width_height(filename: str) -> tuple:
    splitted = filename.split('_')
    return int(splitted[2]), int(splitted[3])

def show_valid_image_during_training(model: nn.Module, image, input_resolution: int, padding_mode: str, device: str) -> Figure:
    padding = ((input_resolution - image.sw) // 2, (input_resolution - image.sh) // 2)
    image_tensor = TF.pad(image.tensor, padding=padding, padding_mode=padding_mode).unsqueeze(0).unsqueeze(0).to(device)
    model = model.to(device)
    preds = model(image_tensor)
    segmentations = dict()
    for i, organ in enumerate(image.organs):
        pred = preds[:, i, :, :]
        segmentation = TF.center_crop(pred, output_size=(image.sh, image.sw))
        segmentations[organ] = segmentation.squeeze(0).detach().cpu().numpy()
    fig = image.show_segmented_images(segmentations=segmentations)
    return fig
