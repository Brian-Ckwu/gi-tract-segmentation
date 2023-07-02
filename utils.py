import yaml
from argparse import Namespace

def load_yaml(file: str) -> Namespace:
    with open(file) as f:
        config = yaml.safe_load(f)
    return Namespace(**config)