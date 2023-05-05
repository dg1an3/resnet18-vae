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

    def encode(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        layer_1_out = self.layer_1(x)
        layer_2_out = self.layer_2(layer_1_out)
        return [layer_1_out, layer_2_out]

    def decode(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        decoder_2_out = self.layer_2.decoder(x)
        decoder_1_out = self.layer_1.decoder(decoder_1_out)
        return [decoder_2_out, decoder_1_out]

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        [layer_1_out, layer_2_out] = self.encode(x)
        [decoder_2_out, decoder_1_out] = self.decode(layer_2_out)
        return [layer_1_out, layer_2_out, decoder_2_out, decoder_1_out]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a random grayscale image (replace this with your actual image data)
    # image = np.random.rand(256, 256)

    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchinfo import summary

    from load_dataset import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # construct the model
    model = V11V(input_size=(1, 448, 448), kernel_size=11, directions=7)
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

    fig, ax = plt.subplots(rows, columns, figsize=(20, 8))
    plt.ion()

    for row in range(rows):
        for column in range(columns):
            filter_weights = model.layer_1.conv_real.weight[row * columns + column][0]
            filter_weights = filter_weights.cpu()
            ax[row][column].imshow(filter_weights, cmap="bone")

    plt.show()

    model = model.to(device)

    # perform training
    train_dataset = load_dataset("cxr8", input_size=(1, 448, 448), clahe_tile_size=8)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    recon_loss_metric = "l1_loss"
    if recon_loss_metric == "binary_cross_entropy":
        recon_loss = lambda x, recon_x: F.binary_cross_entropy(
            x, recon_x, reduction="mean"
        )
    elif recon_loss_metric == "l1_loss":
        recon_loss = F.l1_loss
    elif recon_loss_metric == "mse" or recon_loss_metric != "":
        recon_loss = F.mse

    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    plt.ion()
    num_epochs = 6

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_count = 0
        for batch_idx, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch = model(batch)
            loss_value = recon_loss(batch, recon_batch)

            if train_count % 10 == 0:
                batch = batch.clone().cpu().detach().numpy()
                recon_batch = recon_batch.cpu().detach().numpy()

                # print(v.shape)
                for n in range(5):
                    ax[0][n].imshow(
                        torch.movedim(torch.tensor(batch[n]), 0, -1), cmap="bone"
                    )
                    ax[1][n].imshow(
                        torch.movedim(torch.tensor(recon_batch[n]), 0, -1), cmap="bone"
                    )
            plt.show()
            plt.pause(0.01)

            loss_value.backward()
            train_loss += loss_value.item()
            train_count += 1.0
            optimizer.step()

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch: {batch_idx}, Loss: {train_loss / train_count:.6f} ({loss_value:.6f})"
            )
