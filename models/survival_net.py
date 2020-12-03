# from Multi_Scale_1D_ResNet.model.multi_scale_ori import MSResNet as ResNet1D
# from _3DUnet_Tensorflow_Brats18.train import Unet3dModel as Unet3D

# msresnet = ResNet1D(input_channel=1, layers=[1, 1, 1, 1], num_classes=6)
# msresnet = ResNet1D.cuda()

import torch
from .unet3d import Unet3D
from .resnet1d import ResNet1D

class FeatureMerger(torch.nn.Module):
    def forward(self, x):
        return x

class SurvivalNet(torch.nn.Module):
    def __init__(self, brain_input_channels):
        super().__init__()
        self.unet3D = Unet3D(brain_input_channels)
        self.resnet1D = ResNet1D(1)

        self.fc1 = torch.nn.Linear(20490, 10)
        self.fc2 = torch.nn.Linear(10, 5)

    def forward(self, brain_scan, age_vector):
        brain_scan = self.unet3D(brain_scan)
        # print(brain_scan.size())
        brain_scan = torch.reshape(brain_scan, (brain_scan.size()[0], -1))
        # print(brain_scan.size())

        age_vector = self.resnet1D(age_vector)[0] # get FC layer

        # print(age_vector.size())

        combined_features = torch.cat((brain_scan, age_vector), dim=1)

        # print(combined_features.size())

        combined_features = self.fc1(combined_features)
        combined_features = self.fc2(combined_features)

        return combined_features