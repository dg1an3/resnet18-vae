# -*- coding: utf-8 -*-
"""copyright (c) dglane 2023

Module oriented_powermap.py contains the OrientedPowerMap class, which implements a 
gabor filter bank pytorch module.

It also has a set of unit tests for both OrientedPowerMap and helper functions
"""

import torch.nn as nn


# TODO: switch to use HerringStackLayer in Encoder / Decoder
class HerringStackLayer(nn.Module):
    def __init_(
        self,
        in_channels,
        channels,
        kernels_and_freqs=None,
        loops=3,
        out_res="up-bicubic",
    ):
        pass
