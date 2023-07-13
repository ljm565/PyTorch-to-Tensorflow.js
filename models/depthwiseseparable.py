import torch
import torch.nn as nn
import numpy as np



class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias

        if self.bias:
            self.const = nn.Parameter(torch.ones(self.out_channels, 1) * np.log(1 / 0.07))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, groups=self.in_channels, kernel_size=self.kernel_size, stride=2, padding=self.kernel_size//2, bias=False),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
    
    def forward(self, x):
        if self.bias:
            return self.conv(x) + self.const
        return self.conv(x)
    


class DepthWiseSeparableNet(nn.Module):
    def __init__(self, config):
        super(DepthWiseSeparableNet, self).__init__()
        self.img_size = config.img_size
        self.bias = config.bias

        self.conv1 = nn.Sequential(
            DepthWiseSeparableConv2d(3, 64, 3, self.bias),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            DepthWiseSeparableConv2d(64, 64, 3, self.bias),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = nn.Linear(self.img_size//14 * self.img_size//14 * 128, 2)
        
    
    def forward(self, x):
        batch_size = x.size(0)
        output = self.conv1(x)
        output = self.conv2(output)
        output = output.view(batch_size, -1)
        output = self.fc(output)
        return output