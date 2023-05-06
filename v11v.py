import torch
import torch.nn as nn

from oriented_powermap import OrientedPowerMap


class V11V(nn.Module):
    def __init__(self, in_channels, kernel_size, directions):
        """_summary_

        Args:
            input_size (_type_): _description_
            init_kernel_size (_type_): _description_
            directions (_type_): _description_
        """
        super(V11V, self).__init__()

        self.layer_1 = OrientedPowerMap(
            in_channels=in_channels,
            use_abs=False,
            use_batch_norm=True,
            kernel_size=kernel_size,
            directions=directions,
        )

        self.layer_2 = OrientedPowerMap(
            in_channels=self.layer_1.out_channels,
            use_abs=False,
            use_batch_norm=True,
            kernel_size=kernel_size,
            directions=directions,
        )

        self.layer_3 = OrientedPowerMap(
            in_channels=self.layer_2.out_channels,
            use_abs=False,
            use_batch_norm=True,
            kernel_size=kernel_size,
            directions=directions,
        )

    def encode(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        layer_1_out = self.layer_1(x)
        layer_2_out = self.layer_2(layer_1_out)
        layer_3_out = self.layer_3(layer_2_out)
        return [layer_1_out, layer_2_out, layer_3_out]

    def decode(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        decoder_3_out = self.layer_3.decoder(x)
        decoder_2_out = self.layer_2.decoder(decoder_3_out)
        decoder_1_out = self.layer_1.decoder(decoder_2_out)
        return [decoder_3_out, decoder_2_out, decoder_1_out]

    def forward_dict(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        [layer_1_out, layer_2_out, layer_3_out] = self.encode(x)
        [decoder_3_out, decoder_2_out, decoder_1_out] = self.decode(layer_3_out)
        return {
            "layer_1_out": layer_1_out,
            "layer_2_out": layer_2_out,
            "layer_3_out": layer_3_out,
            "decoder_3_out": decoder_3_out,
            "decoder_2_out": decoder_2_out,
            "decoder_1_out": decoder_1_out,
        }

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        result_dict = self.forward_dict(x)
        return result_dict["decoder_1_out"]


if __name__ == "__main__":
    # construct the model
    model = V11V(in_channels=1, kernel_size=11, directions=7)

    from torchinfo import summary

    print(
        summary(
            model,
            input_size=(37, 1, 448, 448),
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

    # display filters
    columns = 7 + 1  # directions+1
    rows = model.layer_1.conv_real.weight.shape[0] // columns

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(rows, columns, figsize=(10, 4))
    plt.ion()

    for row in range(rows):
        for column in range(columns):
            filter_weights = model.layer_1.conv_real.weight[row * columns + column][0]
            filter_weights = filter_weights.cpu()
            ax[row][column].imshow(filter_weights, cmap="bone")

    plt.show()

    # set up the data loader
    from torch.utils.data import DataLoader
    from cxr8_dataset import get_clahe_transforms, Cxr8Dataset

    import os
    from pathlib import Path
    data_temp_path = os.environ["DATA_TEMP"]
    root_path = Path(data_temp_path) / "cxr8"

    transforms = get_clahe_transforms(clahe_tile_size=8, input_size=448)    
    train_dataset = Cxr8Dataset(root_path, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = model.to(device)

    import torch.nn.functional as F

    recon_loss_metric = "l1_loss"
    if recon_loss_metric == "binary_cross_entropy":
        recon_loss = lambda x, recon_x: F.binary_cross_entropy(
            x, recon_x, reduction="mean"
        )
    elif recon_loss_metric == "l1_loss":
        recon_loss = F.l1_loss
    elif recon_loss_metric == "mse" or recon_loss_metric != "":
        recon_loss = F.mse_loss

    import numpy as np
    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    plt.ion()
    num_epochs = 6

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_count = 0
        for batch_idx, batch_dict in enumerate(train_loader):
            batch_images = batch_dict["image"]
            batch_images = batch_images.to(device)
            optimizer.zero_grad()

            result_dict = model.forward_dict(batch_images)
            recon_images = result_dict["decoder_1_out"]
            # print(batch_images.shape)
            # print(recon_images.shape)
            loss_value = recon_loss(batch_images, recon_images)
            loss_value += recon_loss(result_dict["layer_1_out"], result_dict["decoder_2_out"])

            if train_count % 10 == 0:
                batch_images = batch_images.clone().cpu().detach().numpy()
                recon_images = recon_images.cpu().detach().numpy()

                # print(v.shape)
                for n in range(5):
                    ax[0][n].imshow(np.squeeze(batch_images[n]), cmap="bone")
                    ax[1][n].imshow(np.squeeze(recon_images[n]), cmap="bone")
            plt.show()
            plt.pause(0.01)

            loss_value.backward()
            train_loss += loss_value.item()
            train_count += 1.0
            optimizer.step()

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch: {batch_idx}, Loss: {train_loss / train_count:.6f} ({loss_value:.6f})"
            )
