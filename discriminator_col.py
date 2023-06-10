# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()
        KV = torch.tensor([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]]) / 12.
        #取消修改#修改（3，3，1，1）维度为1，之前是3
        self.KV = Parameter(KV.float().view(1, 1, 5, 5).repeat(3, 3, 1, 1), requires_grad=False)
      
        self.conv1 = SpectralNorm(nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2, bias=False))
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = SpectralNorm(nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=False))
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = SpectralNorm(nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False))
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False))
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = SpectralNorm(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False))
        self.bn5 = nn.BatchNorm2d(128)

        # if WGAN -- 1
        self.fc = nn.Linear(128 * 1 * 1, 1)

    def forward(self, x):
        prep = F.conv2d(x, self.KV, padding=2)

        out = F.tanh(self.bn1(torch.abs(self.conv1(prep))))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.tanh(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn3(self.conv3(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn4(self.conv4(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn5(self.conv5(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    from thop import profile
    net = XuNet()
    print(net)
    x = torch.randn(1, 3, 256, 256)
    print(get_parameter_number(net))
    flops, params = profile(net, inputs=(x,))
    print(flops, params)

