"""
Code adapted from https://medium.com/mlearning-ai/semantic-segmentation-with-pytorch-u-net-from-scratch-502d6565910a
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class CNNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        return self.block(img)

class CNNBlocks(nn.Module):

    def __init__(
        self,
        in_channels: int, # the first in_channels
        out_channels: int,
        num_layers: int = 2, # number of CNN blocks
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(CNNBlock(in_channels, out_channels, kernel_size, stride, padding))
            in_channels = out_channels
    
    def forward(self, img):
        for block in self.blocks:
            img = block(img)
        return img

class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downhills: int = 4, # number of (CNNBlocks + MaxPool2d)
        num_layers: int = 2, # number of CNN blocks per CNNBlocks
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        for _ in range(downhills):
            self.down_blocks.extend(
                [
                    CNNBlocks(in_channels, out_channels, num_layers, kernel_size, stride, padding),
                    nn.MaxPool2d(kernel_size=2) # stride default to 2
                ]
            )
            in_channels = out_channels
            out_channels *= 2
        self.bottom_block = CNNBlocks(in_channels, out_channels, num_layers, kernel_size, stride, padding)
    
    def forward(self, img):
        skip_connections = []
        for module in self.down_blocks:
            img = module(img)
            if isinstance(module, CNNBlocks):
                skip_connections.append(img)
        return self.bottom_block(img), skip_connections

class Decoder(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        n_classes: int, # final out_channels
        uphills: int = 4, # number of (ConvTranpose2D + CNNBlocks)
        num_layers: int = 2, # number of CNN blocks per CNNBlocks
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        for _ in range(uphills):
            out_channels = in_channels // 2
            self.up_blocks.extend(
                [
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    CNNBlocks(in_channels, out_channels, num_layers, kernel_size, stride, padding)
                ]
            )
            in_channels //= 2
        self.segmentation_layer = nn.Conv2d(in_channels, out_channels=n_classes, kernel_size=1, padding=padding)
    
    def forward(self, img, skip_connections: list):
        for module in self.up_blocks:
            if isinstance(module, CNNBlocks):
                connection = skip_connections.pop(-1)
                cropped_connection = TF.center_crop(connection, img.shape[2])
                img = torch.cat([cropped_connection, img], dim=1)
            img = module(img)
        return self.segmentation_layer(img)

class UNet(nn.Module):

    def __init__(
        self,
        n_classes: int,
        first_in_channels: int = 1,
        first_out_channels: int = 64,
        downhills: int = 4,
        uphills: int = 4,
        num_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=first_in_channels,
            out_channels=first_out_channels,
            downhills=downhills,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.decoder = Decoder(
            in_channels=first_out_channels * 2 ** downhills,
            n_classes=n_classes,
            uphills=uphills,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    
    def forward(self, img):
        img, skip_connections = self.encoder(img)
        segmentations = self.decoder(img, skip_connections)
        return segmentations
