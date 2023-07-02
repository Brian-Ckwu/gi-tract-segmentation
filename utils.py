import yaml
from argparse import Namespace

import random
import numpy as np
import torch

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
