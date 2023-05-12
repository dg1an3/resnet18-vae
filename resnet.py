"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .filter_utils import make_oriented_map


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_oriented_phasemap=None):
        super(BasicBlock, self).__init__()
        assert not use_oriented_phasemap
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        use_oriented_maps_bottleneck: Union[str, None] = None,
        oriented_maps_bottleneck_kernel_size: int = 7,
        use_depthwise_maxpool: bool = False,
        use_maxpool_shortcut: bool = False,
    ):
        super(Bottleneck, self).__init__()

        if use_depthwise_maxpool:
            self.maxpool1 = nn.Sequential(
                # use this for depth-wise max pooling
                nn.Unflatten(1, torch.Size([1, in_planes])),
                nn.MaxPool3d(
                    kernel_size=(5, 1, 1), stride=(4, stride, stride), padding=(1, 0, 0)
                ),
                nn.Flatten(start_dim=1, end_dim=2),
                nn.Conv2d(
                    in_planes // 4,
                    planes,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
            )
            # assert False
        else:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, bias=False
            )

        self.bn1 = nn.BatchNorm2d(planes)

        # allow for either phase map or power map
        if "power" in use_oriented_maps_bottleneck:

            conv2_planes_out, self._conv2_real, self._conv2_imag = make_oriented_map(
                in_channels=planes,
                kernel_size=oriented_maps_bottleneck_kernel_size,
                directions=9,
                stride=1,
                dstack_phases=False,
            )

            self.conv2 = lambda x: self._conv2_real(x) ** 2 + self._conv2_imag(x) ** 2

        elif "phase" in use_oriented_maps_bottleneck:

            conv2_planes_out, self.conv2 = make_oriented_map(
                in_channels=planes,
                kernel_size=oriented_maps_bottleneck_kernel_size,
                directions=9,
                stride=1,
                dstack_phases=True,
            )

        else:

            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            conv2_planes_out = planes

        self.bn2 = nn.BatchNorm2d(conv2_planes_out)
        self.conv3 = nn.Conv2d(
            conv2_planes_out, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            assert stride <= 2
            self.shortcut = nn.Sequential(
                # use a MaxPool2d downsampler
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
                if use_maxpool_shortcut
                else nn.Identity(),
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=1 if use_maxpool_shortcut else stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def train_oriented_maps(self, train):
        self.conv2.weight.requires_grad = train
        if hasattr(self, "_conv2_real"):
            self._conv2_real.weight.requires_grad = train
        if hasattr(self, "_conv2_imag"):
            self._conv2_imag.weight.requires_grad = train

    def forward(self, x):
        out = self.maxpool1(x) if hasattr(self, "maxpool1") else self.conv1(x)
        out = F.relu(self.bn1(out))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut_x = self.shortcut(x)
        # print(f"{shortcut_x.shape} vs. {out.shape}")
        out += shortcut_x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet modified to accomodate the Herringstack weights
    """    
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        # trainable_oriented_maps: bool = True,
        use_oriented_maps_v1: Union[str, None] = None,
        oriented_maps_v1_kernel_size: int = 9,
        **kwargs
    ):
        """_summary_

        Args:
            block (_type_): _description_
            num_blocks (_type_): _description_
            num_classes (int, optional): _description_. Defaults to 10.
            use_oriented_maps (_type_, optional): _description_. Defaults to None.
        """
        super(ResNet, self).__init__()

        # set up the initial v1 processing
        if "power" in use_oriented_maps_v1:

            self.in_planes, self._conv1_real, self._conv1_imag = make_oriented_map(
                in_channels=3,
                kernel_size=oriented_maps_v1_kernel_size,
                directions=9,
                stride=1,
            )

            self.conv1 = lambda x: self._conv1_real(x) ** 2 + self._conv1_imag(x) ** 2

        elif "phase" in use_oriented_maps_v1:

            self.in_planes, self.conv1 = make_oriented_map(
                in_channels=3,
                kernel_size=oriented_maps_v1_kernel_size,
                directions=9,
                stride=1,
                dstack_phases=True,
            )

        else:

            self.in_planes = 64
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=1, padding=1, bias=False
            )

        # turn this off for the bottlenecks
        # use_oriented_maps = None

        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            **kwargs,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            **kwargs,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            **kwargs,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            **kwargs,
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # # if the use_oriented_maps parameter is 'init' then we should set to train
        # self.train_oriented_maps(trainable_oriented_maps)

    def train_oriented_maps(self, train=False):
        """_summary_

        Args:
            train (bool, optional): _description_. Defaults to False.
        """
        # TODO: remove this, as we will never train the ori power map
        # if hasattr(self, "_conv1_real"):
        #    self._conv1_real.weight.requires_grad = train
        #    self._conv1_imag.weight.requires_grad = train
        # else:
        #    self.conv1.requires_grad = train

        # recursively set train_oriented_maps
        for child in self.children():
            print(type(child))
            if hasattr(child, "train_oriented_maps"):
                child.train_oriented_maps(train)
            if isinstance(child, nn.Sequential):
                for grandchild in child.modules():
                    print(type(grandchild))
                    if hasattr(grandchild, "train_oriented_maps"):
                        grandchild.train_oriented_maps(train)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        """ """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()