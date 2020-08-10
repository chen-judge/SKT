"""
Student model (1/n-CSRNet) in SKT.
"""
import torch.nn as nn
import torch
from torchvision import models

channel_nums = [[32, 64, 128, 256],  # half
                [21, 43, 85, 171],  # third
                [16, 32, 64, 128],  # quarter
                [13, 26, 51, 102],  # fifth
                ]


class CSRNet(nn.Module):
    def __init__(self, ratio=4, transform=True):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.transform = transform
        channel = channel_nums[ratio-2]
        self.conv0_0 = conv_layers(3, channel[0])
        if self.transform:
            self.transform0_0 = feature_transform(channel[0], 64)
        self.conv0_1 = conv_layers(channel[0], channel[0])

        self.pool0 = pool_layers()
        if transform:
            self.transform1_0 = feature_transform(channel[0], 64)
        self.conv1_0 = conv_layers(channel[0], channel[1])
        self.conv1_1 = conv_layers(channel[1], channel[1])

        self.pool1 = pool_layers()
        if transform:
            self.transform2_0 = feature_transform(channel[1], 128)
        self.conv2_0 = conv_layers(channel[1], channel[2])
        self.conv2_1 = conv_layers(channel[2], channel[2])
        self.conv2_2 = conv_layers(channel[2], channel[2])

        self.pool2 = pool_layers()
        if transform:
            self.transform3_0 = feature_transform(channel[2], 256)
        self.conv3_0 = conv_layers(channel[2], channel[3])
        self.conv3_1 = conv_layers(channel[3], channel[3])
        self.conv3_2 = conv_layers(channel[3], channel[3])

        self.conv4_0 = conv_layers(channel[3], channel[3], dilation=2)
        if transform:
            self.transform4_0 = feature_transform(channel[3], 512)
        self.conv4_1 = conv_layers(channel[3], channel[3], dilation=2)
        self.conv4_2 = conv_layers(channel[3], channel[3], dilation=2)
        self.conv4_3 = conv_layers(channel[3], channel[2], dilation=2)
        if transform:
            self.transform4_3 = feature_transform(channel[2], 256)
        self.conv4_4 = conv_layers(channel[2], channel[1], dilation=2)
        self.conv4_5 = conv_layers(channel[1], channel[0], dilation=2)

        self.conv5_0 = nn.Conv2d(channel[0], 1, kernel_size=1)

        self._initialize_weights()
        self.features = []

    def forward(self, x):
        self.features = []

        x = self.conv0_0(x)
        if self.transform:
            self.features.append(self.transform0_0(x))
        x = self.conv0_1(x)

        x = self.pool0(x)
        if self.transform:
            self.features.append(self.transform1_0(x))
        x = self.conv1_0(x)
        x = self.conv1_1(x)

        x = self.pool1(x)
        if self.transform:
            self.features.append(self.transform2_0(x))
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.pool2(x)
        if self.transform:
            self.features.append(self.transform3_0(x))
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_0(x)
        if self.transform:
            self.features.append(self.transform4_0(x))
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.transform:
            self.features.append(self.transform4_3(x))
        x = self.conv4_4(x)
        x = self.conv4_5(x)

        x = self.conv5_0(x)

        self.features.append(x)

        if self.training is True:
            return self.features
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def conv_layers(inp, oup, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
        nn.ReLU(inplace=True)
    )


def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)


def pool_layers(ceil_mode=True):
    return nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
