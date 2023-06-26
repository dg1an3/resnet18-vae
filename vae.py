# -*- coding: utf-8 -*-
"""copyright (c) dglane 2023

vae.py contains the main VAE class, loss function, and training and inference logic
"""

import os, datetime, logging
from pathlib import Path
from typing import Union

from oriented_powermap import OrientedPowerMap

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


def clamp_01(
    x: Union[dict, torch.Tensor, None], eps: float = 1e-6
) -> Union[dict, torch.Tensor, None]:
    """_summary_

    Args:
        x (Union[dict, torch.Tensor, None]): _description_
        eps (float, optional): _description_. Defaults to 1e-6.

    Returns:
        Union[dict, torch.Tensor, None]: _description_
    """
    match x:
        case None:
            return None

        # if x is dict:
        case dict():
            return {
                key: (
                    value
                    if key in ["mu", "log_var"]
                    else torch.clamp(value, eps, 1.0 - eps)
                )
                for (key, value) in x.items()
            }

        case torch.Tensor():
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
        recon_loss_metrics (Tuple[Func, float]): example: [(F.binary_cross_entropy,0.4),(F.l1_loss,1.0),(F.mse_loss,0.0)]
        beta (_type_): _description_
        x (torch.Tensor): original tensor target to match
        x_v1 (torch.Tensor): upward perceptual loss features. Defaults to None.
        x_v2 (torch.Tensor): upward perceptual loss features. Defaults to None.
        x_v4 (torch.Tensor): upward perceptual loss features. Defaults to None.
        mu (torch.Tensor): mean values for reparameterization
        log_var (torch.Tensor): log variance for reparameterization
        x_v4_back (torch.Tensor): reconstructed perceptual loss features. Defaults to None.
        x_v2_back (torch.Tensor): reconstructed perceptual loss features. Defaults to None.
        x_v1_back (torch.Tensor): reconstructed perceptual loss features. Defaults to None.
        x_back (torch.Tensor): reconstructed value to which to compare

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
    for loss_func, weight in recon_loss_metrics:
        # for value, value_back in [(x, x_back)]:
        recon_loss += loss_func(x, x_back, reduction="mean") * weight

        if x_v1 is not None:
            recon_loss += (
                loss_func(
                    x_v1,
                    x_v1_back,
                    reduction="mean",
                )
                * weight
            )
        if x_v2 is not None:
            recon_loss += (
                loss_func(
                    x_v2,
                    x_v2_back,
                    reduction="mean",
                )
                * weight
            )
        if x_v4 is not None:
            recon_loss += (
                loss_func(
                    x_v4,
                    x_v4_back,
                    reduction="mean",
                )
                * weight
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
        self,
        device,
        input_size,
        init_kernel_size=13,
        latent_dim=32,
        train_stn=False,
        train_non_stn=True,
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

        self.encoder = Encoder(
            device,
            input_size,
            init_kernel_size=init_kernel_size,
            directions=7,
            latent_dim=latent_dim,
        )

        # prepare the STN preprocessor
        # TODO: separate STN in to its own module, so it can be invoked on inputs to:
        #           calculate xform and lut; and apply transform and lut to inputs
        stn_oriented_phasemap_1 = self.encoder.oriented_powermap
        stn_oriented_phasemap_2 = self.encoder.oriented_powermap_2
        stn_oriented_phasemap_3 = self.encoder.oriented_powermap_3
        stn_oriented_phasemap_4 = self.encoder.oriented_powermap_4
        self.localization = nn.Sequential(
            stn_oriented_phasemap_1,
            stn_oriented_phasemap_2,
            stn_oriented_phasemap_3,
            stn_oriented_phasemap_4,
        )

        # for name, param in self.localization.named_parameters():
        # print(f"setting requires grad for {name} to {train_stn}")
        # param.requires_grad = train_stn
        self.localization.to(device)

        # determine size of localization_out
        test_input = torch.randn((1,) + input_size)
        test_input = test_input.to(device)
        localization_out = self.localization(test_input)
        self.localization_out_numel = localization_out.shape.numel()
        print(localization_out.shape)

        self.fc_xform = nn.Sequential(
            nn.Linear(self.localization_out_numel, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 6),
        )

        # initialize to zero weights and biases
        eps = 1e-2
        torch.nn.init.normal_(self.fc_xform[0].weight, 0.0, eps)
        torch.nn.init.normal_(self.fc_xform[0].bias, 0.0, eps)
        torch.nn.init.normal_(self.fc_xform[-1].weight, 0.0, eps)
        torch.nn.init.normal_(self.fc_xform[-1].bias, 0.0, eps)

        for name, param in self.fc_xform.named_parameters():
            print(f"setting requires grad for {name} to {train_stn}")
            param.requires_grad = train_stn

        # determine how many input dimensions to inverse convolve
        dim_to_conv_tranpose = self.encoder.in_planes

        self.decoder = Decoder(
            self.encoder.input_size_to_fc,
            latent_dim=latent_dim,
            out_channels=input_size[0],
            final_kernel_size=init_kernel_size,
            dim_to_conv_tranpose=dim_to_conv_tranpose,
        )

        train_non_stn = True
        for name, param in self.encoder.named_parameters():
            print(f"setting requires grad for {name} to {train_non_stn}")
            param.requires_grad = train_non_stn

        for name, param in self.decoder.named_parameters():
            print(f"setting requires grad for {name} to {train_non_stn}")
            param.requires_grad = train_non_stn

    def stn(self, x):
        logging.debug(f"x.shape = {x.shape}")
        xs = self.localization(x)
        logging.debug(f"xs.shape = {xs.shape}; {'nan' if xs.isnan().any() else ''}")

        xs = xs.view(-1, self.localization_out_numel)
        logging.debug(f"xs.shape = {xs.shape}")

        fc_xform_out = self.fc_xform(xs)

        eps = 0.0

        shear_factor = 0.5
        shear = shear_factor * fc_xform_out[:, 5]
        shear = shear.view(-1, 1)
        shear = torch.clamp(shear, -eps, eps)

        scale_factor = 0.5
        scale_x, scale_y = (
            torch.sigmoid(scale_factor * fc_xform_out[:, 3]) * 2.0,
            torch.sigmoid(scale_factor * fc_xform_out[:, 4]) * 2.0,
        )
        scale_x = scale_x.view(-1, 1)
        scale_y = scale_y.view(-1, 1)

        angle = fc_xform_out[:, 2]
        # angle = torch.clamp(angle, -eps, eps)

        angle_factor = 0.25
        sa = torch.sin(angle_factor * angle).view(-1, 1)
        ca = torch.cos(angle_factor * angle).view(-1, 1)
        # print(f"ca = {ca}")
        # print(f"sa = {sa}")

        xlate_factor = 0.5
        x_shift = xlate_factor * fc_xform_out[:, 0]
        x_shift = x_shift.view(-1, 1)
        # x_shift = torch.clamp(x_shift, -eps, eps)

        y_shift = xlate_factor * fc_xform_out[:, 1]
        y_shift = y_shift.view(-1, 1)
        # y_shift = torch.clamp(y_shift, -eps, eps)

        theta = torch.stack(
            (
                scale_x * ca,
                scale_y * -sa + shear * scale_y * ca,
                -0.5 * (ca - sa) + x_shift + 0.5,
                scale_x * sa,
                scale_y * ca + shear * scale_y * sa,
                -0.5 * (sa + ca) + y_shift + 0.5,
            ),
            dim=-1,
        )
        theta = theta.view(-1, 2, 3)
        # print(f"theta.shape = {theta.shape}")

        # print(f"theta.shape = {theta.shape}; {'nan' if theta.isnan().any() else ''}")
        theta_0 = theta[0].detach().cpu().numpy()
        print(f"theta[0] = {theta_0[0]} {theta_0[1]}")

        # and apply
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode="reflection")  # padding_mode="zeros")

        # TODO: move STN resampling to dataset (with metadata csv)

        return x

    def forward_dict(self, x):
        """perform forward pass, returning a dictionary of useful results for loss functions

        Args:
            x (torch.Tensor): input vector

        Returns:
            dictionary: dictionary of result tensors
        """
        x_stn = self.stn(x)

        # encode the input, returning the gaussian parameters
        result_encoder = self.encoder.forward_dict(x_stn)

        # reparameterization trick!
        z = reparameterize(result_encoder["mu"], result_encoder["log_var"])

        # clamp a subset of latent dimensions
        init_dims = 11
        z = torch.clamp(
            z,
            torch.tensor(
                [-10.0] * init_dims + [0.0] * (self.latent_dim - init_dims)
            ).to(z.device),
            torch.tensor([10.0] * init_dims + [0.0] * (self.latent_dim - init_dims)).to(
                z.device
            ),
        )

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


def load_model(
    input_size,
    device,
    kernel_size=11,
    directions=5,
    latent_dim=96,
    train_stn=False,
    train_non_stn=True,
):
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

    start_epoch = 0
    epoch_files = sorted(list(Path("runs").glob("*_epoch_*.zip")))
    model = VAE(
        device,
        input_size,
        init_kernel_size=kernel_size,
        # directions=directions,
        latent_dim=latent_dim,
        train_stn=train_stn,  # len(epoch_files) >= 0,
        train_non_stn=train_non_stn,
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if len(epoch_files) > 0:
        dct = torch.load(epoch_files[-1], map_location=device)
        start_epoch = dct["epoch"]
        model.load_state_dict(dct["model_state_dict"])
        optimizer.load_state_dict(dct["optimizer_state_dict"])

    logging.info(
        summary(
            model,
            input_size=(3,) + input_size,
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


def train_vae(device, train_stn=False, train_non_stn=True, l1_weight=0.9):
    """perform training of the vae model

    Args:
        device (torch.Device): device to host training
    """
    # TODO: move dataset preparation to cxr8_dataset.py
    data_temp_path = os.environ["DATA_TEMP"]
    root_path = Path(data_temp_path) / "cxr8"

    train_dataset = Cxr8Dataset(
        root_path,
        sz=512,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    input_size = train_dataset[0]["image"].shape
    logging.info(f"input_size = {input_size}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    logging.info(f"train_dataset length = {len(train_dataset)}")

    model, optimizer, start_epoch = load_model(
        input_size,
        device,
        kernel_size=9,
        directions=7,
        latent_dim=96,
        train_stn=train_stn,
        train_non_stn=train_non_stn,
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
        # result_dict["x_v1"] = None
        result_dict["x_v2"] = None
        result_dict["x_v4"] = None
        # result_dict = clamp_01(result_dict)

        recon_loss, kldiv_loss, loss = vae_loss(
            recon_loss_metrics=(
                (F.l1_loss, l1_weight),
                (F.binary_cross_entropy, (1.0 - l1_weight)),
            ),
            beta=0.1,
            x=x,
            **result_dict,
        )

        loss.backward()
        # print(f"loss = {loss}; {'nan' if loss.isnan() else ''}")

        optimizer.step()

        # bit of logic to wait before starting to accumulate loss
        train_count += 1.0 if train_count != -1.0 else 2.0
        train_loss = loss.item() + (train_loss if train_count > 0.0 else 0.0)

        if train_count % 6 == 5:
            x_xform = model.stn(x)

            plot_samples(
                model,
                start_epoch,
                train_loss,
                train_count,
                batch_idx,
                x,
                x_xform,
                result_dict["x_back"],
                recon_loss,
                kldiv_loss,
            )

            # torch.cuda.empty_cache()

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
        f"runs/{log_base}_epoch_{start_epoch+1:02d}.zip",
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

    model, _, start_epoch = load_model(
        input_size, device, kernel_size=7, directions=5, latent_dim=96
    )

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
        train_vae(device, train_stn=True, train_non_stn=True, l1_weight=0.9)
        for _ in range(3):
            for l1_weight in [0.1, 0.9]:
                for train_stn in [False, True]:
                    train_vae(
                        device,
                        train_stn=train_stn,
                        train_non_stn=not (train_stn),
                        l1_weight=l1_weight,
                    )

    if args.infer:
        infer_vae(device, (4, 1024, 1024), args.infer)
