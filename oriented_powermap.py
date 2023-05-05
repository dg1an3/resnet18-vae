import torch
import torch.nn as nn

from filter_utils import make_oriented_map


class OrientedPowerMap(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        in_channels,
        use_abs=False,
        use_batch_norm=False,
        kernel_size=11,
        golden_mean_octaves=[3, 2, 1, 0, -1, -2],
        directions=7,
    ):
        self.in_channels = in_channels

        self.kernel_size = kernel_size
        self.golden_mean_octaves = golden_mean_octaves
        self.directions = directions

        self.use_abs = use_abs
        self.use_batch_norm = use_batch_norm

        kernel_count, weights_real, weights_imag = make_oriented_map(
            inplanes=in_channels,
            kernel_size=self.kernel_size,
            directions=self.directions,
            # golden_mean_octaves=self.golden_mean_octaves,
            stride=1,
        )
        self.out_channels = kernel_count

        self.conv_real = nn.Conv2d(
            in_channels,
            kernel_count,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=False,
        )
        self.conv_real.weight = torch.nn.Parameter(weights_real, requires_grad=False)

        self.conv_imag = nn.Conv2d(
            in_channels,
            kernel_count,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=False,
        )
        self.conv_imag.weight = torch.nn.Parameter(weights_imag, requires_grad=False)

        self.batch_norm = nn.BatchNorm2d(self.out_channels)
        self.activation = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dim_reduce = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=1, bias=False
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.out_channels,
                out_channels=self.in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                output_padding=3,
                bias=False,
            ),
            nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # capture size for later comparison
        pre_size = x.size()

        x = self.conv_real_1(x) ** 2 + self.conv_imag_1(x) ** 2
        if self.use_abs:
            x = torch.sqrt(x)

        if self.use_batch_norm:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.max_pool_2(x)

        x = self.dim_reduce(x)

        # check that sizes are in half
        assert x.size()[0] == pre_size[0] // 2

        return x
