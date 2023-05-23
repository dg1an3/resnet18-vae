import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from functools import reduce
from typing import Union
from filter_utils import make_oriented_map, make_oriented_map_stack_phases

from oriented_powermap import OrientedPowerMap

# TODO: insert [basic_block.py] here
from basic_block import BasicBlock


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        use_oriented_maps_bottleneck: Union[str, None] = None,
        oriented_maps_bottleneck_kernel_size: int = 7,
        use_maxpool_shortcut: bool = False,
    ):
        super(Bottleneck, self).__init__()

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


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        init_kernel_size=9,
        directions=7,
        latent_dim=32,
        use_ori_map="phased",
        use_abs=False,
    ):
        """Resnet-34 based encoder

        Args:
            input_size (torch.Size): _description_
            init_kernel_size (int, optional): _description_. Defaults to 9.
            directions (int, optional): number of equal directions for Gabor filter. Defaults to 7.
            latent_dim (int, optional): number of latent dimensions to convert input to. Defaults to 32.
            use_ori_map (str, optional): these are to be moved to the VAE. Defaults to "phased".
            use_abs (bool, optional): these are to be moved to theVAE. Defaults to False.
        """
        super(Encoder, self).__init__()

        #######################################################################################
        #######################################################################################
        #     ###     ###     ###     ###     ###     ###     ###     ###
        #       ###     ###     ###     ###     ###     ###     ###     ###
        #     ###     ###     ###     ###     ###     ###     ###     ###
        #######################################################################################
        #######################################################################################

        # TODO: move V1 to VAE and reuse V1VxLayer
        logging.warn("constructing oriented_powermap in encoder")
        self.use_abs = use_abs
        self.use_ori_map = use_ori_map
        match self.use_ori_map:
            case "conv_7x7":
                self.conv = nn.Sequential(
                    nn.Conv2d(
                        input_size[0],
                        64,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=True,
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
                self.in_planes = 64
            case "phased":
                self.freq_per_kernel, kernels = make_oriented_map_stack_phases(
                    in_channels=input_size[0],
                    kernel_size=init_kernel_size,
                    directions=directions,
                )
                kernel_count = len(self.freq_per_kernel)
                print(f"len(freq_per_kernel) = {kernel_count}")

                self.freq_per_conv_2_out = None

                conv_1 = nn.Conv2d(
                    input_size[0],
                    kernel_count,
                    kernel_size=init_kernel_size,
                    stride=2,
                    padding=init_kernel_size // 2,
                    bias=True,
                )
                conv_1.weight = torch.nn.Parameter(kernels, requires_grad=False)

                self.conv_2 = nn.Conv2d(
                    in_channels=kernel_count,
                    out_channels=kernel_count // 2,
                    kernel_size=1,
                )

                self.conv = nn.Sequential(
                    conv_1,
                    nn.BatchNorm2d(kernel_count),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    self.conv_2,
                    nn.ReLU(),
                )

                self.in_planes = kernel_count // 2

            case "unphased":
                freq_per_kernel, weights_real_1, weights_imag_1 = make_oriented_map(
                    in_channels=input_size[0],
                    kernel_size=init_kernel_size,
                    directions=directions,
                    # stride=1,
                )

                self.conv_real_1 = nn.Conv2d(
                    input_size[0],
                    len(freq_per_kernel),
                    kernel_size=init_kernel_size,
                    stride=2,
                    padding=init_kernel_size // 2,
                    bias=True,
                )
                self.conv_real_1.weight = torch.nn.Parameter(
                    weights_real_1, requires_grad=False
                )

                self.conv_imag_1 = nn.Conv2d(
                    input_size[0],
                    len(freq_per_kernel),
                    kernel_size=init_kernel_size,
                    stride=2,
                    padding=init_kernel_size // 2,
                    bias=True,
                )
                self.conv_imag_1.weight = torch.nn.Parameter(
                    weights_imag_1, requires_grad=False
                )

                self.post_1 = nn.Sequential(
                    nn.BatchNorm2d(len(freq_per_kernel)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(
                        in_channels=len(freq_per_kernel),
                        out_channels=len(freq_per_kernel),
                        kernel_size=1,
                    ),
                    nn.ReLU(),
                )

                self.in_planes = len(freq_per_kernel)

                print(f"self.in_planes {self.in_planes}")

            case _:
                raise ("unknown")

        #######################################################################################
        #######################################################################################
        #     ###     ###     ###     ###     ###     ###     ###     ###
        #   ###     ###     ###     ###     ###     ###     ###     ###
        #     ###     ###     ###     ###     ###     ###     ###     ###
        #######################################################################################
        #######################################################################################

        self.residual_blocks = nn.Sequential(
            BasicBlock(self.in_planes, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
        )

        # determine output size from v1 + residual blocks
        test_input = torch.randn((1,) + input_size)
        if self.use_ori_map == "unphased":
            v1_output = self.conv_real_1(test_input)
        else:
            v1_output = self.conv(test_input)
        output = self.residual_blocks(v1_output)
        self.input_size_to_fc = output.size()
        print(f"self.input_size_to_fc = {self.input_size_to_fc}")

        self.fc_mu = nn.Linear(self.input_size_to_fc.numel(), latent_dim)
        self.fc_log_var = nn.Linear(self.input_size_to_fc.numel(), latent_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        match self.use_ori_map:
            case "unphased":
                x_conv_real_1 = torch.square(self.conv_real_1(x))
                x_conv_imag_1 = torch.square(self.conv_imag_1(x))
                x = torch.add(x_conv_real_1, x_conv_imag_1)

                if self.use_abs:
                    x = torch.sqrt(x)
                x = self.post_1(x)

                x_after_v1 = x.clone()

            case "phased" | "conv_7x7":
            #     # TODO: calculate freq_per_kernel given current self.conv_2
            #     current_channels = torch.Tensor(self.freq_per_kernel)
            #     current_channels = torch.unsqueeze(current_channels, -1)
            #     current_channels = torch.unsqueeze(current_channels, -1)
            #     current_channels = torch.log(current_channels)
            #     current_channels = current_channels.to(x.device)
            #     self.freq_per_conv_2_out = self.conv_2(current_channels)
            #     self.freq_per_conv_2_out  = torch.exp(self.freq_per_conv_2_out)

                x = self.conv(x)
                x_after_v1 = x.clone()

        x = self.residual_blocks(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var, x_after_v1


if __name__ == "__main__":
    encoder = Encoder((1, 128, 128), 128)
    print(
        summary(
            encoder,
            input_size=(37, 1, 128, 128),
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
    print(set([p.device for p in encoder.parameters()]))
