import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import numpy as np

from filter_utils import make_oriented_map

from encoder import Encoder
from decoder import Decoder


def vae_loss(
    recon_x,
    x,
    mu,
    log_var,
    v1_weight=None,
    x_after_v1=None,
    x_before_v1=None,
    recon_loss_metric="l1_loss",
    beta=1.0,
):
    """_summary_

    Args:
        recon_x (_type_): _description_
        x (_type_): _description_
        mu (_type_): _description_
        log_var (_type_): _description_
        recon_loss_metric (str, optional): _description_. Defaults to 'l1_loss'.
        beta (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    match v1_weight:
        case None:
            pass
        case np.ndarray():
            v1_weight = torch.tensor(v1_weight).to(x_after_v1.device)
        case torch.Tensor():
            v1_weight = v1_weight.to(x_after_v1.device)
        case _:
            raise ("unknown weight type")

    match recon_loss_metric:
        case "binary_cross_entropy":
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")
            if x_after_v1 is not None:
                recon_loss += F.binary_cross_entropy(
                    x_after_v1, x_before_v1, reduction="mean"
                )
        case "l1_loss":
            recon_loss = F.l1_loss(recon_x, x)
            if x_after_v1 is not None:
                if v1_weight is None:
                    recon_loss += F.l1_loss(x_after_v1, x_before_v1, reduction="mean")
                else:
                    v1_loss = F.l1_loss(x_after_v1, x_before_v1, reduction="none")
                    v1_loss = torch.mean(v1_loss, (0, -1, -2))
                    v1_loss = torch.mul(v1_weight, v1_loss)
                    recon_loss += torch.mean(v1_loss)
        case "mse_loss":
            recon_loss = F.mse_loss(recon_x, x)
            if x_after_v1 is not None:
                recon_loss += F.mse_loss(x_after_v1, x_before_v1)
        case _:
            raise ("Unrecognized loss metric")

    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss, kld_loss, recon_loss + beta * kld_loss


class VAE(nn.Module):
    def __init__(self, input_size, init_kernel_size=13, latent_dim=32, use_stn=True):
        """_summary_

        Args:
            input_size (_type_): _description_
            init_kernel_size (int, optional): _description_. Defaults to 11.
            latent_dim (int, optional): _description_. Defaults to 32.
        """
        super(VAE, self).__init__()

        self.use_stn = use_stn
        if self.use_stn:
            # prepare the STN preprocessor

            # TODO: separate STN in to its own module, so it can be invoked on inputs to:
            #           calculate xform and lut; and apply transform and lut to inputs

            self.localization = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 108 * 108, 32), nn.ReLU(True), nn.Linear(32, 2 * 3)
            )
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_size,
            init_kernel_size=init_kernel_size,
            directions=7,
            latent_dim=latent_dim,
            use_ori_map=True,
            use_abs=True,
        )
        self.decoder = Decoder(
            self.encoder.input_size_to_fc,
            latent_dim=latent_dim,
            out_channels=input_size[0],
            final_kernel_size=init_kernel_size,
        )

    def stn(self, x):
        """Spatial transformer network forward function

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # print(x.shape)

        xs = self.localization(x)
        # print(xs.shape)
        if xs.isnan().any(): print(f"xs is nan")

        xs = xs.view(-1, 10 * 108 * 108)
        # print(xs.shape)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        if theta.isnan().any(): print(f"theta is nan")

        # theta[:,:,0] *= 1e-1
        # theta[:,:,1] *= 1e-1
        # theta[:,:,2] *= 1e-1

        # theta[:,0,0] += 1.0
        # theta[:,1,1] += 1.0

        # theta[:,1,2] += 0.05

        print(f"theta[0] = {theta[0].detach().cpu().numpy()}")

        # and apply
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode="reflection")

        return x

    def reparameterize(self, mu, log_var):
        """_summary_

        Args:
            mu (_type_): _description_
            log_var (_type_): _description_

        Returns:
            _type_: _description_
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_dict(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # first apply the transform
        # if self.use_stn:
        x = self.stn(x)

        mu, log_var, x_after_v1 = self.encoder(x)
        # print(f"x_after_v1.shape = {x_after_v1.shape}")
        # print(f"x_after_v2.shape = {x_after_v2.shape}")

        z = self.reparameterize(mu, log_var)

        x_recon, x_before_v1 = self.decoder(z)
        # print(f"x_before_v2.shape = {x_before_v2.shape}")
        # print(f"x_before_v1.shape = {x_before_v1.shape}")

        return {
            "x_recon": x_recon,
            "mu": mu,
            "log_var": log_var,
            "x_after_v1": x_after_v1,
            "x_before_v1": x_before_v1,
        }

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        result_dict = self.forward_dict(x)
        return result_dict["x_recon"]


if "__main__" == __name__:
    argv = "show"
    match argv:
        case "show":
            vae = VAE((1, 224, 224), 128)
            print(
                summary(
                    vae,
                    input_size=(37, 1, 224, 224),
                    depth=10,
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
        case "train":
            pass
        case "test":
            pass
