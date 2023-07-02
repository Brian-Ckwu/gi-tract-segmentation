import yaml
from argparse import Namespace

import random
import numpy as np
import torch
import torch.nn as nn

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
