# -*- coding: utf-8 -*-
"""copyright (c) dglane 2023

module Decoder contains the Decoder class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from functools import reduce

# TODO: insert this directly
from reverse_basic_block import ReverseBasicBlock
from encoder import Encoder


class Decoder(nn.Module):
    """_summary_"""

    def __init__(
        self,
        size_from_fc: torch.Size,
        latent_dim=32,
        out_channels=3,
        final_kernel_size=7,
        dim_to_conv_tranpose=40,
    ):
        """_summary_

        Args:
            size_from_fc (torch.Size): _description_
            latent_dim (int, optional): _description_. Defaults to 32.
            out_channels (int, optional): _description_. Defaults to 3.
            final_kernel_size (int, optional): _description_. Defaults to 7.
            dim_to_conv_tranpose (int, optional): _description_. Defaults to 40.
        """
        super(Decoder, self).__init__()

        self.size_from_fc = size_from_fc
        self.fc = nn.Linear(latent_dim, size_from_fc.numel())

        self.residual_blocks = nn.Sequential(
            ReverseBasicBlock(512, 256, stride=2),
            ReverseBasicBlock(256, 256),
            ReverseBasicBlock(256, 128, stride=2),
            ReverseBasicBlock(128, 128),
            ReverseBasicBlock(128, 64, stride=2),
            ReverseBasicBlock(64, 64),
            ReverseBasicBlock(64, 64, stride=2),
            ReverseBasicBlock(64, dim_to_conv_tranpose),
        )

        self.conv_transpose_1 = nn.Sequential(
            nn.Conv2d(
                dim_to_conv_tranpose,
                out_channels,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2,
                # output_padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.fc(x)
        x = x.view(
            x.size(0),
            self.size_from_fc[-3],
            self.size_from_fc[-2],
            self.size_from_fc[-1],
        )
        x = self.residual_blocks(x)

        x_before_v1 = x.clone()
        x_before_v1 = F.max_pool2d(x_before_v1, kernel_size=3, stride=2, padding=1)

        x = self.conv_transpose_1(x)

        return x, x_before_v1


if "__main__" == __name__:
    encoder = Encoder((1, 128, 128), 128)
    decoder = Decoder(128, encoder.input_size_to_fc, 1)
    print(
        summary(
            decoder,
            input_size=(
                37,
                128,
            ),
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
