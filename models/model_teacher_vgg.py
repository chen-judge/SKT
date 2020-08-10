"""
Teacher model in SKT
"""
import torch.nn as nn
import torch
from torchvision import models
from utils import save_net, load_net, cal_para


class CSRNet(nn.Module):
    def __init__(self, pretrained=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()
        self.features = []
        if pretrained:
            print 'load vgg pretrained model'
            mod = models.vgg16(pretrained=True)
            for i in xrange(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self, x):
        self.features = []
        # frontend: VGG
        x = self.frontend(x)
        # backend: dilated convolution
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def regist_hook(self):
        self.features = []

        def get(model, input, output):
            # function will be automatically called each time, since the hook is injected
            self.features.append(output.detach())

        for name, module in self._modules['frontend']._modules.items():
            if name in ['1', '4', '9', '16']:
                self._modules['frontend']._modules[name].register_forward_hook(get)
        for name, module in self._modules['backend']._modules.items():
            if name in ['1', '7']:
                self._modules['backend']._modules[name].register_forward_hook(get)


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