import torch.nn as nn
import torch
from abc import ABC, abstractmethod
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class Discriminator(nn.Module):
    def __init__(self, inputc) -> None:
        '''
        PatchGan Discriminator
        '''
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(inputc, 64, kernel_size=4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # layer 1
        x = self.relu(self.conv1(x))
        # layer 2, 3, 4
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        # layer 5
        x = self.conv5(x)
        return x