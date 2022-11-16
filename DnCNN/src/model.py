import torch
from torch import nn


class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_channels=3, num_features=64):
        super(DnCNN, self).__init__()

        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_features = num_features

        self.layers = []

        self.layers.append(nn.Conv2d(self.num_channels, self.num_features,
                                     kernel_size=3, padding=1))
        self.layers.append(nn.ReLU())

        for i in range(self.num_layers - 2):
            self.layers.append(nn.Conv2d(self.num_features, self.num_features,
                                         kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(num_features=self.num_features))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Conv2d(self.num_features, self.num_channels,
                                     kernel_size=3, padding=1))

        self.dncnn = nn.Sequential(*self.layers)

    def forward(self, img):
        output = self.dncnn(img)
        return img - output
