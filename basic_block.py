"""
copyright (c) 2023, dglane

This module defines the BasicBlock class, which is a custom PyTorch module that represents a basic building block for constructing deeper neural networks, particularly in the context of Residual Networks (ResNets).

The BasicBlock consists of two convolutional layers, each followed by batch normalization and a ReLU activation function. It also includes a shortcut connection to facilitate gradient flow and learning in deeper models.

Example:
    from torch import nn
    from basic_block import BasicBlock
    
    # Create a BasicBlock instance with specific input and output channels
    block = BasicBlock(in_channels=64, out_channels=128, stride=2)
    
    # Use the BasicBlock instance in a custom neural network
    class CustomResNet(nn.Module):
        def __init__(self):
            super(CustomResNet, self).__init__()
            self.layer1 = BasicBlock(3, 64)
            self.layer2 = BasicBlock(64, 128, stride=2)
            # ...
    
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            # ...
            return x
"""
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class BasicBlock(nn.Module):
    """
    BasicBlock is a PyTorch module implementing a basic building block for Residual Networks (ResNets).

    This block consists of two convolutional layers, each followed by batch normalization and ReLU activation.
    It also includes a shortcut connection to help with gradient flow and learning deeper models.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initializes a BasicBlock instance.

        Args:
            in_channels (int): The number of input channels for the first convolutional layer.
            out_channels (int): The number of output channels for both convolutional layers.
            stride (int, optional): The stride for the first convolutional layer. Defaults to 1.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=True
                )
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        """
        Computes the output of the BasicBlock given an input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        identity = self.shortcut(identity)

        out += identity
        out = F.relu(out)
        return out


if __name__ == "__main__":
    print(
        summary(
            BasicBlock(16, 32, 1),
            input_size=(37, 16, 128, 128),
            col_names=[
                "input_size",
                "kernel_size",
                "mult_adds",
                "num_params",
                "output_size",
                "trainable",
            ],
        )
    )
