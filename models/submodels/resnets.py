import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, resnet34, resnet50


from models import ENCODERS

@ENCODERS.register_module('resnet50')
class MyResNet50(nn.Module):
    def __init__(self):
        super(MyResNet50, self).__init__()
        self.module = resnet50()
        self.module.fc = nn.Identity()

    def forward(self, x):
        return self.module(x)


@ENCODERS.register_module('resnet34')
class MyResNet34(nn.Module):
    def __init__(self):
        super(MyResNet34, self).__init__()
        self.module = resnet34()
        self.module.fc = nn.Identity()

    def forward(self, x):
        return self.module(x)


@ENCODERS.register_module('resnet18')
class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()
        self.module = resnet18()
        self.module.fc = nn.Identity()

    def forward(self, x):
        return self.module(x)


if __name__ == '__main__':
    print(ENCODERS)
    net = ENCODERS['resnet18']()
    print(net)
    x = torch.from_numpy(np.zeros([2, 3, 224, 224])).float()
    print(net(x).shape)