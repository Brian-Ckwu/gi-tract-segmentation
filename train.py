from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from argparse import Namespace
from data import GIImage, GIImageDataset, GIImageDataLoader, train_valid_split_cases
from utils import load_yaml, set_seed
from model import UNet
from evaluation import evaluate

def train(config: Namespace):
    set_seed(config.seed)
    
    model = UNet(n_classes=len(GIImage.organs))
    
    train_cases, valid_cases = train_valid_split_cases(config.input_path, config.valid_size)
    train_set = GIImageDataset(image_path=config.input_path, label_path=config.label_path, cases=train_cases)
    valid_set = GIImageDataset(image_path=config.input_path, label_path=config.label_path, cases=valid_cases)

    train_loader = GIImageDataLoader(
        model=model,
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        input_resolution=config.input_resolution,
        padding_mode=config.padding_mode
    ).get_data_loader()

    valid_loader = GIImageDataLoader(
        model=model,
        dataset=valid_set,
        batch_size=config.batch_size,
        shuffle=False,
        input_resolution=config.input_resolution,
        padding_mode=config.padding_mode
    ).get_data_loader()
    
    # TODO: use other loss functions and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)
    
    model = model.to(config.device)

    best_valid_loss = float("inf") # TODO: track model performance with other metrics
    for epoch in range(config.nepochs):
        model.train()
        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch {epoch + 1}")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            preds = model(inputs)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update()
            
            if (i % config.valid_steps == 0) or (i == len(train_loader) - 1):
                total_valid_loss = evaluate(model, valid_loader, criterion, config.device)
                print(f"Valid loss: {total_valid_loss:.4f}")
                # save model if validation loss is improved
                if total_valid_loss < best_valid_loss:
                    best_valid_loss = total_valid_loss
                    torch.save(model.state_dict(), Path(config.save_path) / "model.pth")
                    (Path(config.save_path) / "valid_loss.txt").write_text(f"{total_valid_loss:.4f}")
                    print(f"Model saved at {config.save_path}")

if __name__ == "__main__":
    config = load_yaml("config.yml")
    train(config)
