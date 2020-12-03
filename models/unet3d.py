import torch
import torch.nn.functional as F

class Unet3D(torch.nn.Module):
    """ first half of unet """
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = Unet3dBlock(in_channels)
        self.block2 = Unet3dBlock(in_channels*4)
        self.block3 = Unet3dBlock(in_channels*16, final=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class Unet3dBlock(torch.nn.Module):
    def __init__(self, channels, final=False):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(
            channels, channels*2,
            kernel_size=3,
            stride=1
        )

        self.batch_norm1 = BN_Relu(channels*2)

        self.conv2 = torch.nn.Conv3d(
            channels*2, channels*4,
            kernel_size=3,
            stride=1
        )

        self.batch_norm2 = BN_Relu(channels*4)

        self.final = final

        if not self.final:
            self.downsize = torch.nn.Conv3d(
                channels*4, channels*4,
                kernel_size=3,
                stride=2
            )

            self.batch_norm_downsize = BN_Relu(channels*4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)

        if not self.final:
            x = self.downsize(x)
            x = self.batch_norm_downsize(x)
        
        return x

class BN_Relu(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = torch.nn.BatchNorm3d(num_features) # OR InstanceNorm5d()

    def forward(self, x):
        return F.relu(self.norm(x))
