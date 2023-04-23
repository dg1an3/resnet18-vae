import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from filter_utils import make_oriented_map

from encoder import Encoder
from decoder import Decoder


def vae_loss(recon_x, x, mu, log_var, recon_loss_metric="l1_loss", beta=1.0):
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
    if recon_loss_metric == "binary_cross_entropy":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")
    elif recon_loss_metric == "l1_loss":
        recon_loss = F.l1_loss(recon_x, x)
    elif recon_loss_metric == "mse" or recon_loss_metric != "":
        recon_loss = F.mse(recon_x, x)

    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss, kld_loss, recon_loss + beta * kld_loss


class VAE(nn.Module):
    def __init__(self, input_size, init_kernel_size=11, latent_dim=32):
        """_summary_

        Args:
            input_size (_type_): _description_
            init_kernel_size (int, optional): _description_. Defaults to 11.
            latent_dim (int, optional): _description_. Defaults to 32.
        """
        super(VAE, self).__init__()

        # prepare the STN preprocessor
        #
        directions = 7
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
            stride=1,
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
            stride=1,
            padding=init_kernel_size // 2,
            bias=False,
        )
        self.conv_imag.weight = torch.nn.Parameter(
            weights_imag, requires_grad=False
        )
         
        self.use_abs = False
        self.localization = nn.Sequential(
            # nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(kernel_count, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 110 * 110, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_size,
            init_kernel_size=init_kernel_size,
            directions=7,
            latent_dim=latent_dim,
            use_ori_map=True,
            use_abs=False,
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
        print(x.shape)

        x_ = self.conv_real(x) ** 2 + self.conv_imag(x) ** 2
        print(x_.shape)
        if self.use_abs:
            x_ = torch.sqrt(x_)
        xs = self.localization(x_)
        print(xs.shape)
        xs = xs.view(-1, 10 *110 * 110)
        # print(xs.shape)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

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

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # first apply the transform
        x = self.stn(x)

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var


if "__main__" == __name__:
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
