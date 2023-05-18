import numpy as np
from torchinfo import summary


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
