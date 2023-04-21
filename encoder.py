import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchsummary

from functools import reduce
from filter_utils import make_oriented_map

from basic_block import BasicBlock


class Encoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_size,
        init_kernel_size=9,
        directions=7,
        latent_dim=32,
        use_ori_map=True,
        use_abs=False,
    ):
        super(Encoder, self).__init__()

        self.use_abs = use_abs
        self.use_ori_map = use_ori_map
        if use_ori_map:
            kernel_count, weights_real, weights_imag = make_oriented_map(
                inplanes=input_size[0],
                kernel_size=init_kernel_size,
                directions=directions,
                stride=1,
            )

            self.conv_real = nn.Conv2d(
                input_size[0],
                kernel_count,
                kernel_size=init_kernel_size,
                stride=2,
                padding=init_kernel_size // 2,
                bias=False,
            )
            self.conv_real.weight = torch.nn.Parameter(
                weights_real, requires_grad=False
            )

            self.conv_imag = nn.Conv2d(
                input_size[0],
                kernel_count,
                kernel_size=init_kernel_size,
                stride=2,
                padding=init_kernel_size // 2,
                bias=False,
            )
            self.conv_imag.weight = torch.nn.Parameter(
                weights_imag, requires_grad=False
            )

            self.post = nn.Sequential(
                nn.BatchNorm2d(kernel_count),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.in_planes = kernel_count

        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    input_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.in_planes = 64

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

        fixed_size = False
        if fixed_size:
            self.input_size_to_fc = [512, 16, 16]
        else:
            self.input_size_to_fc = (
                torchsummary.summary(
                    nn.Sequential(
                        self.conv_real if self.use_ori_map else self.conv,
                        self.post,
                        self.residual_blocks,
                    ),
                    input_size,
                    verbose=0,
                )
                .summary_list[-1]
                .output_size[1:]
            )
            print(f"self.input_size_to_fc = {self.input_size_to_fc}")

        self.inputs_to_fc = reduce(lambda x, y: x * y, self.input_size_to_fc)

        self.fc_mu = nn.Linear(self.inputs_to_fc, latent_dim)
        self.fc_log_var = nn.Linear(self.inputs_to_fc, latent_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.use_ori_map:
            x = self.conv_real(x) ** 2 + self.conv_imag(x) ** 2
            if self.use_abs:
                x = torch.sqrt(x)
            x = self.post(x)
        else:
            x = self.conv(x)

        x = self.residual_blocks(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


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
