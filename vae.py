import os, datetime, logging
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")

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


def vae_loss(
    recon_loss_metrics,
    beta,
    x,
    mu,
    log_var,
    x_recon,
    v1_weight=None,
    perc_loss=None,
    perc_loss_recon=None,
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
    x_recon = clamp_01(x_recon)
    perc_loss = clamp_01(perc_loss)
    perc_loss_recon = clamp_01(perc_loss_recon)

    if isinstance(v1_weight, np.ndarray):
        v1_weight = torch.tensor(v1_weight).to(perc_loss.device)

    if isinstance(v1_weight, torch.Tensor):
        v1_weight = v1_weight.to(perc_loss.device)
    elif v1_weight is not None:
        raise ("unknown weight type")

    recon_loss = 0.0
    for loss_func in recon_loss_metrics:
        recon_loss += loss_func(x, x_recon, reduction="mean")
        if perc_loss is not None:
            use_weight = (
                v1_weight is not None # and loss_func is not F.binary_cross_entropy
            )
            v1_loss = loss_func(
                perc_loss,
                perc_loss_recon,
                reduction="none" if use_weight else "mean",
            )
            if use_weight:
                v1_loss = torch.mean(v1_loss, (0, -1, -2))
                v1_loss = torch.mul(v1_weight, v1_loss)
                v1_loss = torch.mean(v1_loss)

            recon_loss += v1_loss

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


class VAE(nn.Module):
    def __init__(self, input_size, init_kernel_size=13, latent_dim=32, use_stn=True):
        """construct a resnet 34 VAE module

        Args:
            input_size (torch.Size): input size for the model
            init_kernel_size (int, optional): sz x sz of kernel. Defaults to 11.
            latent_dim (int, optional): latent dimension of VAE. Defaults to 32.
        """
        super(VAE, self).__init__()

        # construct a dummy input tensor
        # input = torch.randn(input_size)

        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_size,
            init_kernel_size=init_kernel_size,
            directions=7,
            latent_dim=latent_dim,
            use_ori_map="phased",  # means that there are 80 dimensions
            use_abs=True,
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
        # set up the v1 weight for weighted loss function
        if self.encoder.freq_per_conv_2_out is not None:
            v1_weight = torch.Tensor(self.encoder.freq_per_conv_2_out)
        else:
            # TODO: get this from self.encoder(x)
            phi = (5**0.5 + 1) / 2  # golden ratio
            v1_weight = [
                weight for n in range(1, -4, -1) for weight in [phi ** (n * 1)] * 8
            ]
            v1_weight = torch.Tensor(list(v1_weight))
        v1_weight = None  # need to determine how to calculate weight, given the conv1x1

        # encode the input, returning the gaussian parameters
        mu, log_var, perc_loss = self.encoder(x)

        # reparameterization trick!
        z = reparameterize(mu, log_var)

        # and decode back to the original
        x_recon, perc_loss_recon = self.decoder(z)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "log_var": log_var,
            "perc_loss": perc_loss,
            "perc_loss_recon": perc_loss_recon,
            "v1_weight": v1_weight,
        }

    def forward(self, x):
        """computes the model for a given input x

        Args:
            x (torch.Tensor): input tensor, generally as batches of input_size

        Returns:
            torch.Tensor: reconstructed x for the given input
        """
        result_dict = self.forward_dict(x)
        return result_dict["x_recon"]


def load_model(input_size, device, kernel_size=13, directions=5, latent_dim=96):
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
        latent_dim=latent_dim,
        use_stn=True,
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
            input_size=(37, input_size[0], input_size[1], input_size[2]),
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

    return model, optimizer, start_epoch


def plot_samples(
    model,
    start_epoch,
    train_loss,
    train_count,
    batch_idx,
    x,
    x_recon,
    recon_loss,
    kldiv_loss,
):
    """_summary_

    Args:
        model (_type_): _description_
        start_epoch (_type_): _description_
        train_loss (_type_): _description_
        train_count (_type_): _description_
        batch_idx (_type_): _description_
        x (_type_): _description_
        x_recon (_type_): _description_
        recon_loss (_type_): _description_
        kldiv_loss (_type_): _description_
    """
    fig, ax = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(
        f"Epoch {start_epoch+1} Batch {batch_idx} Loss: {train_loss / train_count:.6f} ({recon_loss:.6f}/{kldiv_loss:.6f})"
    )
    fig.patch.set_facecolor("xkcd:gray")

    # fig.show()
    # TODO: move this to output to tensorboard
    x = x[0:5].clone()

    x_xformed = x  # model.stn(x) if model.use_stn else x
    x_xformed = x_xformed.detach().cpu().numpy()

    x = x.detach().cpu().numpy()

    x_recon = x_recon[0:5].clone()
    x_recon = x_recon.detach().cpu().numpy()

    # additive blending
    blend_data = np.stack([x_recon, x_xformed, x_recon], axis=-1)
    blend_data = np.clip(blend_data, a_min=0.0, a_max=1.0)

    # print(v.shape)
    for n in range(5):
        ax[0][n].imshow(np.squeeze(x[n]), cmap="bone")
        ax[1][n].imshow(np.squeeze(blend_data[n]))  # cmap='bone')
        ax[2][n].imshow(np.squeeze(x_recon[n]), cmap="bone")

    fig.tight_layout()
    fig.savefig(f"runs/{log_base}_current.png")
    plt.close(fig)


def train_vae(device):
    """perform training of the vae model

    Args:
        device (torch.Device): device to host training
    """
    # TODO: move dataset preparation to another function
    data_temp_path = os.environ["DATA_TEMP"]
    root_path = Path(data_temp_path) / "cxr8"

    # TODO: do we still need transforms
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

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    logging.info(f"train_dataset length = {len(train_dataset)}")

    model, optimizer, start_epoch = load_model(input_size, device)
    logging.info(set([p.device for p in model.parameters()]))

    # torch.autograd.set_detect_anomaly(True)

    # DONE: only execute single epoch
    model.train()
    train_loss = 0
    train_count = -10
    for batch_idx, batch in enumerate(train_loader):
        x = batch["image"].to(device)

        optimizer.zero_grad()

        result_dict = model.forward_dict(x)
        if result_dict["v1_weight"] is not None:
            result_dict["v1_weight"] = result_dict["v1_weight"].to(device)
            logging.info(f"v1_weight = {result_dict['v1_weight']}")

        recon_loss, kldiv_loss, loss = vae_loss(
            recon_loss_metrics=(F.binary_cross_entropy, F.l1_loss),
            beta=0.2,
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
                result_dict["x_recon"],
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


def infer_vae(device, input_size, source_dir):
    """_summary_

    Args:
        device (_type_): _description_
        input_size (_type_): _description_
        source_dir (_type_): _description_
    """
    print(f"inferring images in {source_dir}")

    model, _, start_epoch = load_model(input_size, device)

    print(
        summary(
            model,
            input_size=(37,) + input_size,
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
        infer_vae(device, (1, 448, 448), args.infer)
