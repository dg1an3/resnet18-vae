# -*- coding: utf-8 -*-
"""copyright (c) dglane 2023

encoder.py contains the Encoder class
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from typing import Union
from filter_utils import make_oriented_map

from oriented_powermap import OrientedPowerMap

# TODO: insert [basic_block.py] here
from basic_block import BasicBlock


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(
#         self,
#         in_planes,
#         planes,
#         stride=1,
#         use_oriented_maps_bottleneck: Union[str, None] = None,
#         oriented_maps_bottleneck_kernel_size: int = 7,
#         use_maxpool_shortcut: bool = False,
#     ):
#         super(Bottleneck, self).__init__()

#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=1, stride=stride, bias=False
#         )

#         self.bn1 = nn.BatchNorm2d(planes)

#         # allow for either phase map or power map
#         if "power" in use_oriented_maps_bottleneck:
#             conv2_planes_out, self._conv2_real, self._conv2_imag = make_oriented_map(
#                 in_channels=planes,
#                 kernel_size=oriented_maps_bottleneck_kernel_size,
#                 directions=9,
#                 stride=1,
#                 dstack_phases=False,
#             )

#             self.conv2 = lambda x: self._conv2_real(x) ** 2 + self._conv2_imag(x) ** 2

#         elif "phase" in use_oriented_maps_bottleneck:
#             conv2_planes_out, self.conv2 = make_oriented_map(
#                 in_channels=planes,
#                 kernel_size=oriented_maps_bottleneck_kernel_size,
#                 directions=9,
#                 stride=1,
#                 dstack_phases=True,
#             )

#         else:
#             self.conv2 = nn.Conv2d(
#                 planes, planes, kernel_size=3, stride=1, padding=1, bias=False
#             )
#             conv2_planes_out = planes

#         self.bn2 = nn.BatchNorm2d(conv2_planes_out)
#         self.conv3 = nn.Conv2d(
#             conv2_planes_out, self.expansion * planes, kernel_size=1, bias=False
#         )
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             assert stride <= 2
#             self.shortcut = nn.Sequential(
#                 # use a MaxPool2d downsampler
#                 nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
#                 if use_maxpool_shortcut
#                 else nn.Identity(),
#                 nn.Conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=1 if use_maxpool_shortcut else stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )

#     def train_oriented_maps(self, train):
#         self.conv2.weight.requires_grad = train
#         if hasattr(self, "_conv2_real"):
#             self._conv2_real.weight.requires_grad = train
#         if hasattr(self, "_conv2_imag"):
#             self._conv2_imag.weight.requires_grad = train

#     def forward(self, x):
#         out = self.maxpool1(x) if hasattr(self, "maxpool1") else self.conv1(x)
#         out = F.relu(self.bn1(out))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         shortcut_x = self.shortcut(x)
#         # print(f"{shortcut_x.shape} vs. {out.shape}")
#         out += shortcut_x
#         out = F.relu(out)
#         return out


#######################################################################################
#######################################################################################
#     ###     ###     ###     ###     ###     ###     ###     ###
#       ###     ###     ###     ###     ###     ###     ###     ###
#     ###     ###     ###     ###     ###     ###     ###     ###
#######################################################################################
#######################################################################################


class Encoder(nn.Module):
    def __init__(
        self,
        device,
        input_size,
        init_kernel_size=9,
        directions=7,
        latent_dim=32,
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

        # TODO: move V1 to VAE and reuse V1VxLayer
        logging.info("constructing oriented_powermap in encoder")

        self.oriented_powermap = OrientedPowerMap(
            device,
            input_size[0],
            kernel_size=init_kernel_size,
            frequencies=None,
            directions=directions,
            use_powermap=False,
        )
        self.oriented_powermap.to(device)

        self.oriented_powermap_2 = OrientedPowerMap(
            device,
            self.oriented_powermap.out_channels,
            kernel_size=init_kernel_size,
            frequencies=None,
            directions=directions,
            out_res="/2",
        )
        self.oriented_powermap_2.to(device)

        self.oriented_powermap_3 = self.oriented_powermap_2
        self.oriented_powermap_4 = self.oriented_powermap_2

        # self.freq_per_kernel = self.oriented_powermap.freq_per_kernel
        self.in_planes = self.oriented_powermap_3.out_channels

        self.residual_blocks = nn.Sequential(
            # 64x64
            OrientedPowerMap(
                device,
                in_channels=self.in_planes,
                out_channels=64,
                kernel_size=7,
                frequencies=None,
                out_res=None,
            ),  # BasicBlock(self.in_planes, 64, stride=2),
            OrientedPowerMap(
                device,
                in_channels=64,
                out_channels=64,
                kernel_size=7,
                frequencies=None,
                out_res=None,
            ),  # BasicBlock(64, 64),
            OrientedPowerMap(
                device,
                in_channels=64,
                out_channels=64,
                kernel_size=7,
                frequencies=None,
                out_res=None,
            ),  # BasicBlock(64, 64),
            OrientedPowerMap(
                device,
                in_channels=64,
                out_channels=64,
                kernel_size=7,
                frequencies=None,
                out_res="/2",  # TODO: move to final step in each layer ; then to separate operation
            ),
            # 128x128
            # OrientedPowerMap(
            #     device,
            #     in_channels=64,
            #     out_channels=128,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res=None,
            # ),
            # OrientedPowerMap(
            #     device,
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res=None,
            # ),
            # OrientedPowerMap(
            #     device,
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res=None,
            # ),
            # OrientedPowerMap(
            #     device,
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res=None,
            # ),
            # OrientedPowerMap(
            #     device,
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res=None,
            # ),
            # OrientedPowerMap(
            #     device,
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res="/2",
            # ),
            # 256x256
            # OrientedPowerMap(
            #     device,
            #     in_channels=128,
            #     out_channels=256,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res=None,
            # ),
            # OrientedPowerMap(
            #     device,
            #     in_channels=256,
            #     out_channels=256,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res=None,
            # ),
            # OrientedPowerMap(
            #     device,
            #     in_channels=256,
            #     out_channels=256,
            #     kernel_size=7,
            #     frequencies=None,
            #     out_res="/2",
            # ),
        )
        self.residual_blocks.to(device)

        self.penultimate_conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
        ).to(device)

        self.penultimate_residual = OrientedPowerMap(
            device,
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            frequencies=None,
            out_res=None,
        ).to(device)
        self.penultimate_decimate = nn.AvgPool2d(kernel_size=3, stride=2, padding=1).to(
            device
        )

        self.final_conv1 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=1,
        ).to(device)

        self.final_residual = OrientedPowerMap(
            device,
            in_channels=256,
            out_channels=256,
            kernel_size=7,
            frequencies=None,
            out_res=None,
        ).to(device)
        self.final_decimate = nn.AvgPool2d(kernel_size=3, stride=2, padding=1).to(
            device
        )

        # determine output size from v1 + residual blocks
        test_input = torch.randn((1,) + input_size)
        test_input = test_input.to(device)

        v1_output = self.oriented_powermap(test_input)
        v1_output = self.oriented_powermap_2(v1_output)
        v1_output = self.oriented_powermap_3(v1_output)
        # v1_output = self.oriented_powermap_4(v1_output)

        output = self.residual_blocks(v1_output)

        output = self.penultimate_conv1(output)
        penultimate_bypass = torch.clone(output)
        for _ in range(6):
            output = self.penultimate_residual(output)
        output = 0.5 * (penultimate_bypass + output)            
        output = self.penultimate_decimate(output)

        output = self.final_conv1(output)
        final_bypass = torch.clone(output)
        for _ in range(3):
            output = self.final_residual(output)
        output = 0.5 * (final_bypass + output)
        output = self.final_decimate(output)        

        self.input_size_to_fc = output.size()
        print(f"self.input_size_to_fc = {self.input_size_to_fc}")

        self.fc_mu = nn.Linear(self.input_size_to_fc.numel(), latent_dim)
        self.fc_log_var = nn.Linear(self.input_size_to_fc.numel(), latent_dim)

    def forward_dict(self, x):
        """perform forward pass and accumulate intermediate results

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_v1 = self.oriented_powermap(x)
        x_v2 = self.oriented_powermap_2(x_v1)
        x_v4 = self.oriented_powermap_3(x_v2)

        x = self.residual_blocks(x_v4)

        # perform last two residual block
        # TODO: move this to OrientedPowerMap
        x = self.penultimate_conv1(x)
        for _ in range(6):
            x = self.penultimate_residual(x)
        x = self.penultimate_decimate(x)

        x = self.final_conv1(x)
        for _ in range(3):
            x = self.final_residual(x)
        x = self.final_decimate(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return {"x_v1": x_v1, "x_v2": x_v2, "x_v4": x_v4, "mu": mu, "log_var": log_var}

    def forward(self, x):
        """calculate forward encoder

        Args:
            x (torch.Tensor): input tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mean tensor, log_variance tensor, post-perceptual response)
        """
        result_dict = self.forward_dict(x)
        return result_dict["mu"], result_dict["log_var"]


#######################################################################################
#######################################################################################
#     ###     ###     ###     ###     ###     ###     ###     ###
#   ###     ###     ###     ###     ###     ###     ###     ###
#     ###     ###     ###     ###     ###     ###     ###     ###
#######################################################################################
#######################################################################################


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
