import torch
import torch.nn as nn

from tqdm import tqdm
from data import GIImageDataLoader

# TODO: other metrics
def evaluate(model: nn.Module, data_loader: GIImageDataLoader, criterion: nn.Module, device: str, use_fp16: bool) -> float:
    model.eval()
    model = model.to(device)
    criterion = criterion.to(device)
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                preds = model(inputs)
                loss = criterion(preds, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss
