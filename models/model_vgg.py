"""
For training CSRNet teacher
"""
import torch.nn as nn
import torch
from torchvision import models
# from utils import save_net,load_net
import time


class CSRNet(nn.Module):
    def __init__(self, pretrained=True):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        # cal_para(self.frontend)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if pretrained:
            self._initialize_weights(mode='normal')
            mod = models.vgg16(pretrained=True)
            state_keys = list(self.frontend.state_dict().keys())
            pretrain_keys = list(mod.state_dict().keys())
            for i in range(len(self.frontend.state_dict().items())):
                # self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
                # print(mod.state_dict()[pretrain_keys[i]])
                self.frontend.state_dict()[state_keys[i]].data = mod.state_dict()[pretrain_keys[i]].data
        else:
            self._initialize_weights(mode='kaiming')
                
    def forward(self, x):
        # front relates to VGG
        x = self.frontend(x)
        # backend relates to dilated convolution
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == 'normal':
                    nn.init.normal_(m.weight, std=0.01)
                elif mode == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

