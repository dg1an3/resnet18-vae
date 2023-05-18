import sys, os, datetime, logging
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
from torchinfo import summary


# from filter_utils import make_oriented_map

from cxr8_dataset import Cxr8Dataset, get_clahe_transforms

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

    x = torch.clamp(x,  1e-6, 1.0-1e-6)
    recon_x = torch.clamp(recon_x, 1e-6, 1.0-1e-6)
    if x_after_v1 is not None:
        x_after_v1 = torch.clamp(x_after_v1,  1e-6, 1.0-1e-6)
        x_before_v1 = torch.clamp(x_before_v1,  1e-6, 1.0-1e-6)

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
            recon_loss += F.l1_loss(recon_x, x)
            #recon_loss = F.kl_div(x_recon, x, reduction="batchmean")
            if x_after_v1 is not None:
                recon_loss += F.binary_cross_entropy(
                     x_after_v1, x_before_v1, reduction="mean"
                )
                recon_loss += F.l1_loss(x_after_v1, x_before_v1, reduction="mean")
                #recon_loss += F.kl_div(x_before_v1, x_after_v1, reduction="batchmean")
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

        input = torch.randn(input_size)

        self.use_stn = use_stn
        if self.use_stn:
            # prepare the STN preprocessor

            # TODO: separate STN in to its own module, so it can be invoked on inputs to:
            #           calculate xform and lut; and apply transform and lut to inputs

            self.localization = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(6, 12, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                # nn.Conv2d(16, 32, kernel_size=5),
                # nn.MaxPool2d(2, stride=2),
                # nn.ReLU(True),
            )

            # determine size of localization_out
            localization_out = self.localization(input)
            self.localization_out_numel = localization_out.shape.numel()
            # print(localization_out.shape, )

            self.fc_loc = nn.Sequential(
                nn.Linear(self.localization_out_numel, 32),
                nn.ReLU(True),
                nn.Linear(32, 2 * 3),
            )
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
            use_ori_map="phased",
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
        logging.debug(f"x.shape = {x.shape}")

        xs = self.localization(x)
        logging.debug(f"xs.shape = {xs.shape}; {'nan' if xs.isnan().any() else ''}")

        xs = xs.view(-1, self.localization_out_numel)
        logging.debug(f"xs.shape = {xs.shape}")

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        logging.debug(f"theta.shape = {theta.shape}")

        use_clamp = True
        if use_clamp:
            theta[:, 0, 0] = torch.clamp(theta[:, 0, 0].clone(), 0.999, 1.001)
            theta[:, 1, 1] = torch.clamp(theta[:, 1, 1].clone(), 0.999, 1.001)

            theta[:, 0, 1] = torch.clamp(theta[:, 0, 1].clone(), -0.001, 0.001)
            theta[:, 1, 0] = torch.clamp(theta[:, 1, 0].clone(), -0.001, 0.001)

            theta[:, :, 2] = torch.clamp(theta[:, :, 2].clone(), -0.2, 0.2)

        use_shift_center = False
        if use_shift_center:
            # convert theta to be centered on normalized image coordinates
            shift_center = torch.tensor(
                [[[1, 0, -0.5], [0, 1, -0.5]]], dtype=torch.float
            )
            logging.debug(f"shift_center.shape = {shift_center.shape}")

            shift_center = shift_center.to(theta.device)
            theta = torch.matmul(
                theta,
                shift_center,
            )
            theta = torch.matmul(
                torch.tensor([[1, 0, 0.5], [0, 1, 0.5]], dtype=torch.float).to(
                    theta.device
                ),
                theta,
            )

        # print(f"theta.shape = {theta.shape}; {'nan' if theta.isnan().any() else ''}")
        theta_0 = theta[0].detach().cpu().numpy()
        logging.debug(f"theta[0] = {theta_0[0]} {theta_0[1]}")

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
        if self.use_stn:
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
    log_base = datetime.date.today().strftime("%Y%m%d")
    logging.basicConfig(
        filename=f"runs/{log_base}_vae_main.log",
        format="%(asctime)s|%(levelname)s|%(module)s|%(funcName)s|%(message)s",
        level=logging.DEBUG,
    )
    match sys.argv:
        case [_, "train"]:
            data_temp_path = os.environ["DATA_TEMP"]
            root_path = Path(data_temp_path) / "cxr8"

            transforms = get_clahe_transforms(clahe_tile_size=8, input_size=448)
            train_dataset = Cxr8Dataset(root_path, transform=transforms)

            input_size = train_dataset[0]["image"].shape
            logging.info(f"input_size = {input_size}")

            train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
            logging.info(f"train_dataset length = {len(train_dataset)}")

            KERNEL_SIZE = 11
            DIRECTIONS = 5
            LATENT_DIM = 96  # 64

            # model = VAE((1 if dataset_name == "cxr8" else 3, 224, 224), latent_dim).to(device)
            model = VAE(
                input_size,
                init_kernel_size=KERNEL_SIZE,
                latent_dim=LATENT_DIM,
                use_stn=True,
            )
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

            device = torch.device("cuda" if torch.cuda.is_available else "cpu")
            logging.info(device)

            model = model.to(device)
            logging.info(set([p.device for p in model.parameters()]))

            num_epochs = 3

            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            # TODO: get this from the VAE construction
            v1_weight = torch.tensor(
                [1.0**1] * 8
                + [0.5**1] * 8
                + [0.25**1] * 8
                + [0.125**1] * 8
                + [0.06125**1] * 8
            )
            logging.info(f"v1_weight = {v1_weight}")

            # torch.autograd.set_detect_anomaly(True)

            start_epoch = 0
            epoch_files = sorted(list(Path("runs").glob("*_epoch_*.zip")))
            if len(epoch_files) > 0:
                dct = torch.load(epoch_files[-1])
                start_epoch = dct["epoch"]
                model.load_state_dict(dct["model_state_dict"])
                optimizer.load_state_dict(dct["optimizer_state_dict"])

            for epoch in range(start_epoch, start_epoch + num_epochs):
                model.train()
                train_loss = 0
                train_count = 0
                for batch_idx, batch in enumerate(train_loader):
                    x = batch["image"]

                    x = x.to(device)
                    optimizer.zero_grad()

                    result_dict = model.forward_dict(x)
                    x_recon = result_dict["x_recon"]
                    # TODO: can these be passed as **params?
                    recon_loss, kldiv_loss, loss = vae_loss(
                        x,
                        x_recon,
                        result_dict["mu"],
                        result_dict["log_var"],
                        v1_weight,
                        result_dict["x_after_v1"],
                        result_dict["x_before_v1"],
                        recon_loss_metric="binary_cross_entropy",
                        # recon_loss_metric="l1_loss",
                        beta=0.1,
                    )

                    if train_count % 10 == 9:
                        fig, ax = plt.subplots(3, 5, figsize=(20, 12))
                        fig.suptitle(
                            f"Epoch {epoch+1}/{start_epoch + num_epochs} Batch {batch_idx}"
                        )
                        fig.patch.set_facecolor("xkcd:gray")

                        # fig.show()
                        # TODO: move this to output to tensorboard
                        x = x[0:5].clone()

                        x_xformed = model.stn(x) if model.use_stn else x
                        x_xformed = x_xformed.detach().cpu().numpy()

                        x = x.detach().cpu().numpy()

                        x_recon = x_recon[0:5].clone()
                        x_recon = x_recon.detach().cpu().numpy()

                        # additive blending
                        blend_data = np.stack([x_recon, x_xformed, x_recon], axis=-1)

                        # print(v.shape)
                        for n in range(5):
                            ax[0][n].imshow(np.squeeze(x[n]), cmap="bone")
                            ax[1][n].imshow(np.squeeze(blend_data[n]))  # cmap='bone')
                            ax[2][n].imshow(np.squeeze(x_recon[n]), cmap="bone")

                        # drawing updated values
                        # fig.canvas.draw()

                        # This will run the GUI event
                        # loop until all UI events
                        # currently waiting have been processed
                        # fig.canvas.flush_events()

                        fig.tight_layout()
                        fig.savefig(f"runs/{log_base}_current.png")
                        plt.close(fig)

                    loss.backward()
                    # print(f"loss = {loss}; {'nan' if loss.isnan() else ''}")

                    train_loss += loss.item()
                    train_count += 1.0

                    # print(list(model.parameters()))
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    logging.info(f"Epoch {epoch+1} of {num_epochs}: Batch {batch_idx}")
                    logging.info(
                        f"Loss: {train_loss / train_count:.6f} ({recon_loss:.6f}/{kldiv_loss:.6f})"
                    )

                logging.info(f"saving model for epoch {epoch+1}")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"runs/{log_base}_epoch_{epoch+1}.zip",
                )

            logging.info("completed training")

        case [_, "infer"]:
            pass

        case [_]:
            # TODO: print help
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
