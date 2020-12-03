import torch


class Unet3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
