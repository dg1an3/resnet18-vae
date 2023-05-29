# -*- coding: utf-8 -*-
"""copyright (c) dglane 2023

vae.py contains the main VAE class, loss function, and training and inference logic
"""

import os, datetime, logging
from pathlib import Path

import numpy as np

from show_utils import plot_samples

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchinfo import summary


from cxr8_dataset import Cxr8Dataset

from encoder import Encoder
from decoder import Decoder


def clamp_01(x, eps=1e-6):
    if x is None:
        return None

    return torch.clamp(x, eps, 1.0 - eps)


#############################################################################################################
##########################################
############################
#########
####


def vae_loss(
    recon_loss_metrics,
    beta,
    x,
    x_v1,
    x_v2,
    x_v4,
    mu,
    log_var,
    x_v4_back,
    x_v2_back,
    x_v1_back,
    x_back,
):
    """compute the total VAE loss, including reconstruction error and kullback-liebler divergence with a unit gaussian

    Args:
        recon_loss_metrics (Tuple[Func]): example: (F.binary_cross_entropy,F.l1_loss,F.mse_loss)
        x (torch.Tensor): original tensor target to match
        mu (torch.Tensor): mean values for reparameterization
        log_var (torch.Tensor): log variance for reparameterization
        x_recon (torch.Tensor): reconstructed value to which to compare
        v1_weight (torch.Tensor, optional): weight vector for perceptual loss. Defaults to None.
        perc_loss (torch.Tensor, optional): upward perceptual loss features. Defaults to None.
        perc_loss_recon (torch.Tensor, optional): reconstructed perceptual loss features. Defaults to None.

    Returns:
        Tuple[float,float,float]: (reconstruction loss, kldiv loss, total loss)
    """

    x = clamp_01(x)
    x_back = clamp_01(x_back)
    x_v1 = clamp_01(x_v1)
    x_v1_back = clamp_01(x_v1_back)
    x_v2 = clamp_01(x_v2)
    x_v2_back = clamp_01(x_v2_back)
    x_v4 = clamp_01(x_v4)
    x_v4_back = clamp_01(x_v4_back)

    recon_loss = 0.0
    for loss_func in recon_loss_metrics:
        recon_loss += loss_func(x, x_back, reduction="mean")
        if x_v1 is not None:
            recon_loss += loss_func(
                x_v1,
                x_v1_back,
                reduction="mean",
            )
        if x_v2 is not None:
            recon_loss += loss_func(
                x_v2,
                x_v2_back,
                reduction="mean",
            )
        if x_v4 is not None:
            recon_loss += loss_func(
                x_v4,
                x_v4_back,
                reduction="mean",
            )

    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss, kld_loss, recon_loss + beta * kld_loss


def reparameterize(mu, log_var):
    """perform reparameterization trick given mean and log variance

    Args:
        mu (torch.Tensor): mean tensor for gaussian
        log_var (torch.Tensor): log variances for gaussian

    Returns:
        torch.Tensor: sampled value
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


##############################################################################################
######################################
###################
#########
####


class VAE(nn.Module):
    def __init__(
        self, input_size, init_kernel_size=13, latent_dim=32, use_v1_weights=False
    ):
        """construct a resnet 34 VAE module

        Args:
            input_size (torch.Size): input size for the model
            init_kernel_size (int, optional): sz x sz of kernel. Defaults to 11.
            latent_dim (int, optional): latent dimension of VAE. Defaults to 32.
        """
        super(VAE, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.use_v1_weights = use_v1_weights

        self.encoder = Encoder(
            input_size,
            init_kernel_size=init_kernel_size,
            directions=5,
            latent_dim=latent_dim,
        )

        # determine how many input dimensions to inverse convolve
        dim_to_conv_tranpose = self.encoder.in_planes

        self.decoder = Decoder(
            self.encoder.input_size_to_fc,
            latent_dim=latent_dim,
            out_channels=input_size[0],
            final_kernel_size=init_kernel_size,
            dim_to_conv_tranpose=dim_to_conv_tranpose,
        )

    def forward_dict(self, x):
        """perform forward pass, returning a dictionary of useful results for loss functions

        Args:
            x (torch.Tensor): input vector

        Returns:
            dictionary: dictionary of result tensors
        """
        # encode the input, returning the gaussian parameters
        result_encoder = self.encoder.forward_dict(x)

        # reparameterization trick!
        z = reparameterize(result_encoder["mu"], result_encoder["log_var"])

        # and decode back to the original
        result_decoder = self.decoder.forward_dict(z)

        return {**result_encoder, **result_decoder}

    def forward(self, x):
        """computes the model for a given input x

        Args:
            x (torch.Tensor): input tensor, generally as batches of input_size

        Returns:
            torch.Tensor: reconstructed x for the given input
        """
        result_dict = self.forward_dict(x)
        return result_dict["x_back"]


####
#########
############################
#########
####
####
#########
############################
#########
####
####
#########
############################
#########
####
####
#########
############################
#########
####


def load_model(input_size, device, kernel_size=11, directions=5, latent_dim=96):
    """_summary_

    Args:
        input_size (_type_): _description_
        device (_type_): _description_
        kernel_size (int, optional): _description_. Defaults to 11.
        directions (int, optional): _description_. Defaults to 5.
        latent_dim (int, optional): _description_. Defaults to 96.

    Returns:
        _type_: _description_
    """
    model = VAE(
        input_size,
        init_kernel_size=kernel_size,
        # directions=directions,
        latent_dim=latent_dim,
        use_v1_weights=False,
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    epoch_files = sorted(list(Path("runs").glob("*_epoch_*.zip")))
    if len(epoch_files) > 0:
        dct = torch.load(epoch_files[-1], map_location=device)
        start_epoch = dct["epoch"]
        model.load_state_dict(dct["model_state_dict"])
        optimizer.load_state_dict(dct["optimizer_state_dict"])

    logging.info(
        summary(
            model,
            input_size=(3, input_size[0], input_size[1], input_size[2]),
            depth=10,
            col_names=[
                "input_size",
                "kernel_size",
                # "mult_adds",
                # "num_params",
                "output_size",
                "trainable",
            ],
        )
    )

    torch.cuda.empty_cache()

    return model, optimizer, start_epoch


####
#########
#################
#########################
###########################################


def train_vae(device):
    """perform training of the vae model

    Args:
        device (torch.Device): device to host training
    """
    # TODO: move dataset preparation to cxr8_dataset.py
    data_temp_path = os.environ["DATA_TEMP"]
    root_path = Path(data_temp_path) / "cxr8"

    train_dataset = Cxr8Dataset(
        root_path,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    input_size = train_dataset[0]["image"].shape
    logging.info(f"input_size = {input_size}")

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    logging.info(f"train_dataset length = {len(train_dataset)}")

    model, optimizer, start_epoch = load_model(
        input_size, device, kernel_size=7, directions=5, latent_dim=96
    )
    logging.info(set([p.device for p in model.parameters()]))

    # torch.autograd.set_detect_anomaly(True)

    # DONE: only execute single epoch
    model.train()
    train_loss = 0
    train_count = -10

    # release from this batch
    torch.cuda.empty_cache()
    for batch_idx, batch in enumerate(train_loader):
        x = batch["image"].to(device)

        optimizer.zero_grad()

        result_dict = model.forward_dict(x)

        recon_loss, kldiv_loss, loss = vae_loss(
            recon_loss_metrics=(F.binary_cross_entropy, F.l1_loss),
            beta=0.05,
            x=x,
            **result_dict,
        )

        loss.backward()
        # print(f"loss = {loss}; {'nan' if loss.isnan() else ''}")

        optimizer.step()

        # bit of logic to wait before starting to accumulate loss
        train_count += 1.0 if train_count != -1.0 else 2.0
        train_loss = loss.item() + (train_loss if train_count > 0.0 else 0.0)

        if train_count % 10 == 9:
            plot_samples(
                model,
                start_epoch,
                train_loss,
                train_count,
                batch_idx,
                x,
                result_dict["x_back"],
                recon_loss,
                kldiv_loss,
            )

        logging.info(f"Epoch {start_epoch+1}: Batch {batch_idx}")
        logging.info(
            f"Loss: {train_loss / train_count:.6f} ({recon_loss:.6f}/{kldiv_loss:.6f})"
        )

        # release from this batch
        # torch.cuda.empty_cache()

    logging.info(f"saving model for epoch {start_epoch+1}")
    torch.save(
        {
            "epoch": start_epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"runs/{log_base}_epoch_{start_epoch+1}.zip",
    )

    logging.info("completed training")


####
#########
############################
##########################################
#############################################################################################################


def infer_vae(device, input_size, source_dir):
    """_summary_

    Args:
        device (_type_): _description_
        input_size (_type_): _description_
        source_dir (_type_): _description_
    """
    print(f"inferring images in {source_dir}")

    model, _, start_epoch = load_model(input_size, device, kernel_size=7, directions=5, latent_dim=96)

    print(
        summary(
            model,
            input_size=(37,) + input_size,
            depth=10,
            col_names=[
                "input_size",
                "kernel_size",
                # "mult_adds",
                # "num_params",
                "output_size",
                "trainable",
            ],
        )
    )

    # TODO: use bokeh to create a figure for inferences
    logging.warn("TODO: use bokeh to create a figure for inferences")


if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", help="perform a single epoch of training"
    )
    parser.add_argument(
        "--infer",
        type=str,
        help="performance inference on images in specified directory",
    )
    args = parser.parse_args()

    # TODO: log config via yaml
    logging.warn("TODO: switch to log config via yaml")
    log_base = datetime.date.today().strftime("%Y%m%d")
    logging.basicConfig(
        filename=f"runs/{log_base}_vae_main.log",
        format="%(asctime)s|%(levelname)s|%(module)s|%(funcName)s|%(message)s",
        level=logging.DEBUG,
    )

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    logging.info(f"torch operations on {device} device")

    if args.train:
        train_vae(device)

    if args.infer:
        infer_vae(device, (1, 1024, 1024), args.infer)
