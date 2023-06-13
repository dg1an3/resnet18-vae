# -*- coding: utf-8 -*-
"""copyright (c) dglane 2023

filter_utils.py implements a gabor pyramid for pytorch.
"""

import datetime
import numpy as np
from torchinfo import summary
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")


def show_summary(model, input_size):
    """_summary_

    Args:
        model (_type_): _description_
        input_size (_type_): _description_
    """
    columns_to_show = [
        "input_size",
        "kernel_size",
        "mult_adds",
        "num_params",
        "output_size",
        "trainable",
    ]
    print(
        summary(
            model,
            input_size=input_size,
            depth=10,
            col_names=columns_to_show,
        )
    )


def plot_data(model, x, x_recon, axes):
    # TODO: move this to output to tensorboard
    x = x[0:5].clone()
    x_recon = x_recon[0:5].clone()

    if model.use_stn:
        x_xformed = model.stn(x)
    else:
        x_xformed = x

    x = x.detach().cpu().numpy()
    x_recon = x_recon.detach().cpu().numpy()
    x_xformed = x_xformed.detach().cpu().numpy()

    # additive blending
    blend_data = np.stack([x_recon, x_xformed, x_recon], axis=-1)

    # print(v.shape)
    for n in range(5):
        axes[0][n].imshow(np.squeeze(x[n]), cmap="bone")
        axes[1][n].imshow(np.squeeze(blend_data[n]))  # cmap='bone')
        axes[2][n].imshow(np.squeeze(x_recon[n]), cmap="bone")


def plot_samples(
    model,
    start_epoch,
    train_loss,
    train_count,
    batch_idx,
    x,
    x_xform,
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

    log_base = datetime.date.today().strftime("%Y%m%d")
    fig, ax = plt.subplots(3, 6, figsize=(20, 12))
    fig.suptitle(
        f"Epoch {start_epoch+1} Batch {batch_idx} Loss: {train_loss / train_count:.6f} ({recon_loss:.6f}/{kldiv_loss:.6f})"
    )
    fig.patch.set_facecolor("xkcd:gray")

    # fig.show()
    # TODO: move this to output to tensorboard
    x = x.detach().cpu()
    x = x[0:5].clone()
    x = x.numpy()

    x_xform = x_xform.detach().cpu()
    x_xform = x_xform[0:5].clone()
    x_xform = x_xform.numpy()

    x_recon = x_recon.detach().cpu()
    x_recon = x_recon[0:5].clone()
    x_recon = x_recon.numpy()

    # additive blending
    blend_data_1 = np.stack(
        [x_recon[:, 1, ...], x_xform[:, 1, ...], x_recon[:, 1, ...]], axis=-1
    )
    blend_data_1 = np.clip(blend_data_1, a_min=0.0, a_max=1.0)

    blend_data_2 = np.stack(
        [x_recon[:, 2, ...], x_xform[:, 2, ...], x_recon[:, 2, ...]], axis=-1
    )
    blend_data_2 = np.clip(blend_data_2, a_min=0.0, a_max=1.0)

    blend_data_3 = np.stack(
        [x_recon[:, 3, ...], x_xform[:, 3, ...], x_recon[:, 3, ...]], axis=-1
    )
    blend_data_3 = np.clip(blend_data_3, a_min=0.0, a_max=1.0)

    # print(v.shape)
    for n in range(2):
        ax[0][n * 3].imshow(np.squeeze(x[n, 1, ...]), vmin=0.0, vmax=1.0, cmap="bone")
        ax[1][n * 3].imshow(np.squeeze(blend_data_1[n]))  # cmap='bone')
        ax[2][n * 3].imshow(
            np.around(400.0 * np.squeeze(x_recon[n, 1, ...]), decimals=0),
            vmin=0.0,
            vmax=400.0,
            cmap="bone",
        )
        ax[0][n * 3 + 1].imshow(
            np.squeeze(x[n, 2, ...]), vmin=0.0, vmax=1.0, cmap="bone"
        )
        ax[1][n * 3 + 1].imshow(np.squeeze(blend_data_2[n]))  # cmap='bone')
        ax[2][n * 3 + 1].imshow(
            np.around(400.0 * np.squeeze(x_recon[n, 2, ...]), decimals=0),
            vmin=0.0,
            vmax=400.0,
            cmap="bone",
        )
        ax[0][n * 3 + 2].imshow(
            np.squeeze(x[n, 3, ...]), vmin=0.0, vmax=1.0, cmap="bone"
        )
        ax[1][n * 3 + 2].imshow(np.squeeze(blend_data_3[n]))  # cmap='bone')
        ax[2][n * 3 + 2].imshow(
            np.around(400.0 * np.squeeze(x_recon[n, 3, ...]), decimals=0),
            vmin=0.0,
            vmax=400.0,
            cmap="bone",
        )

    fig.tight_layout()
    fig.savefig(f"runs/{log_base}_current.png")
    plt.close(fig)
