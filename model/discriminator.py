import torch.nn as nn
import torch
from abc import ABC, abstractmethod

class Discriminator(nn.Module):
    def __init__(self, inputc) -> None:
        '''
        PatchGan Discriminator
        '''
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(inputc, 64, kernel_size=4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # layer 1
        x = self.relu(self.conv1(x))        # shape: 
        # layer 2, 3, 4
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        # layer 5
        x = self.relu(self.bn5(self.conv5(x)))
        # layer 6
        x = self.conv6(x)
        return x
