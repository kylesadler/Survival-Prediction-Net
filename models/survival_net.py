# from Multi_Scale_1D_ResNet.model.multi_scale_ori import MSResNet as ResNet1D
# from _3DUnet_Tensorflow_Brats18.train import Unet3dModel as Unet3D

# msresnet = ResNet1D(input_channel=1, layers=[1, 1, 1, 1], num_classes=6)
# msresnet = ResNet1D.cuda()

import torch
from .unet3d import Unet3D
from .resnet1D import ResNet1D

class FeatureMerger(torch.nn.Module):
    def forward(self, x)

class SurvivalNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet3D = Unet3D
        self.resnet1D = ResNet1D

        self.fc1 = torch.nn.Linear(9216, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, brain_scan, age_vector):
        brain_scan = self.unet3D(brain_scan)
        brain_scan = torch.flatten(brain_scan)

        age_vector = self.resnet1D(age_vector)


        combined_features = torch.cat((brain_scan, age_vector), dim=0)

        combined_features = self.fc1(combined_features)
        combined_features = self.fc2(combined_features)

        return combined_features