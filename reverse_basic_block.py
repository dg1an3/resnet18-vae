import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class ReverseBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ReverseBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            # output_padding=0,  # stride - 1,
            bias=True,
        )
        self.upsample1 = nn.Upsample(scale_factor=stride, mode="bilinear")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    # output_padding=0,  # stride - 1,
                    bias=True,
                ),
                nn.Upsample(scale_factor=stride, mode="bilinear"),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.upsample1(self.conv1(x))))
        out = self.bn2(self.conv2(out))

        identity = self.shortcut(identity)

        out += identity
        out = F.relu(out)
        return out


if "__main__" == __name__:
    print(
        summary(
            ReverseBasicBlock(16, 32, 1),
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
