'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

from meta_module import *
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        
        x = gamma * x + beta
        
        return x

class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', lora_dim=32):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if True:
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(MetaModule):
    def __init__(self, block, num_blocks, num_classes=10, lora_dim=16):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, lora_dim=lora_dim)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, lora_dim=lora_dim)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, lora_dim=lora_dim)
        self.linear = MetaLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, lora_dim=16):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, lora_dim=lora_dim))
            self.in_planes = planes * block.expansion

        return MetaSequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet8(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1], num_classes, **kwargs)

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


from exp_utils import rsetattr
class ResNetWrapper(MetaModule):
    def __init__(self, resnet, P):
        super().__init__()
        self.resnet = resnet
        self.P = P
        self.init_weight = dict(resnet.named_parameters())
        self.theta = MetaLinear(40, 1, bias=False)
        nn.init.zeros_(self.theta.weight)
    def forward(self, x):
        #self.resnet.load_state_dict(self.init_weight)
        grads = torch.mm(self.P.transpose(0, 1), self.theta.weight).reshape(-1)
        idx = 0
        for (name, p), po in zip(self.resnet.named_parameters(), self.init_weight.values()):
            rsetattr(self.resnet, name, po - grads[idx:idx + p.nelement()].view(p.shape))
            idx = idx + p.nelement()
        assert idx == self.P.shape[1]

        return self.resnet(x)

if __name__ == "__main__":
    a = resnet8(num_classes=10)
    inpu = torch.randn(1,3,32,32)
    a(inpu)

