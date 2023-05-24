# -*- coding: utf-8 -*-
"""copyright (c) dglane 2023

Module oriented_powermap.py contains the OrientedPowerMap class, which implements a 
gabor filter bank pytorch module.

It also has a set of unit tests for both OrientedPowerMap and helper functions
"""

import torch
import torch.nn as nn

# TODO: insert filter_utils here, and include unit tests
from filter_utils import make_oriented_map, make_oriented_map_stack_phases


class OrientedPowerMap(nn.Module):
    """This is an implementation of an oriented (gabor) powermap convolutional layer in pytorch.

    The oriented powermap can be used either
     * preserving phase: (directions+1) x frequencies x 2 phases filters will be generated
     * phaseless: in which case (directions+1) x frequencies x 1 response magnitude filters
        will be generated

    It is a reasonable approximation to V1-like receptive fields.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size=11,
        frequencies=None,
        directions=7,
    ):
        """construct an OrientedPowerMap

        Args:
            in_channels (int): number of input channels
            kernel_size (int, optional): _description_. Defaults to 11.
            golden_mean_octaves (list, optional): list of octaves of the golden mean to be
                represented. Defaults to [3, 2, 1, 0, -1, -2].
            directions (int, optional): number of dimensions. Defaults to 7.
        """
        super(OrientedPowerMap, self).__init__()

        self.in_channels = in_channels

        if True:  # new
            self.kernel_size = kernel_size
            if frequencies:
                self.frequencies = frequencies
            else:
                self.frequencies = [2.0, 1.0, 0.5, 0.25, 0.125]

            self.directions = directions

            self.freq_per_kernel, kernels = make_oriented_map_stack_phases(
                in_channels=in_channels,
                kernel_size=kernel_size,
                directions=directions,
            )
            kernel_count = len(self.freq_per_kernel)
            print(f"len(freq_per_kernel) = {kernel_count}")

            conv_1 = nn.Conv2d(
                in_channels,
                kernel_count,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
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
            self.out_channels = self.conv_2.out_channels

        else:  # old
            kernel_count, weights_real, weights_imag = make_oriented_map(
                in_channels=in_channels,
                kernel_size=self.kernel_size,
                directions=self.directions,
                frequencies=self.frequencies,
                stride=1,
            )

            self.conv_real = nn.Conv2d(
                in_channels,
                kernel_count,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                bias=False,
            )
            self.conv_real.weight = torch.nn.Parameter(
                weights_real, requires_grad=False
            )

            self.conv_imag = nn.Conv2d(
                in_channels,
                kernel_count,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                bias=False,
            )
            self.conv_imag.weight = torch.nn.Parameter(
                weights_imag, requires_grad=False
            )

            self.out_channels = self.conv_real.weight.shape[0] // 10

            self.batch_norm = nn.BatchNorm2d(self.conv_real.weight.shape[0])
            self.activation = nn.ReLU()
            self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.dim_reduce = nn.Conv2d(
                self.conv_real.weight.shape[0],
                self.out_channels,
                kernel_size=1,
                bias=False,
            )

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(
        #         self.out_channels,
        #         out_channels=self.in_channels,
        #         kernel_size=self.kernel_size,
        #         stride=1,
        #         padding=self.kernel_size // 2,
        #         # output_padding=0,
        #         bias=False,
        #     ),
        #     nn.Upsample(scale_factor=2, mode="bilinear"),
        #     nn.BatchNorm2d(self.in_channels),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # capture size for later comparison
        #?pre_size = x.size()

        x = self.conv(x)

        # check that sizes are in half
        # print(f"pre_size = {pre_size}; post_size = {x.size()}")
        # assert x.size()[-1] == pre_size[-1] // 2

        return x


#######################################################################################
#######################################################################################
#     ###     ###     ###     ###     ###     ###     ###     ###
#     ###     ###     ###     ###     ###     ###     ###     ###
#     ###     ###     ###     ###     ###     ###     ###     ###
#######################################################################################
#######################################################################################

import unittest
from torchinfo import summary


class TestOrientedPowerMap(unittest.TestCase):
    def test_torchinfo(self):
        # construct the model
        model = OrientedPowerMap(in_channels=1)
        model = model.to("cpu")

        print(
            summary(
                model,
                input_size=(37, 1, 448, 448),
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

    def test_encode_decode(self):
        # construct the model
        model = OrientedPowerMap(in_channels=1)
        model = model.to("cpu")

        x = torch.randn((47, 1, 448, 448)).to("cuda")
        x_latent = model(x)
        x_prime = model.decoder(x_latent)
        print(x, x_latent, x_prime)


if __name__ == "__main__":
    unittest.main()
