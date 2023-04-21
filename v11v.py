import torch
import torch.nn as nn
from torchinfo import summary

from filter_utils import make_oriented_map


class V11V(nn.Module):
    def __init__(self, input_size, init_kernel_size, directions):
        """_summary_

        Args:
            input_size (_type_): _description_
            init_kernel_size (_type_): _description_
            directions (_type_): _description_
        """
        super(V11V, self).__init__()
        self.use_abs = False

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
            stride=2,
            padding=init_kernel_size // 2,
            bias=False,
        )
        self.conv_real.weight = torch.nn.Parameter(weights_real, requires_grad=False)

        self.conv_imag = nn.Conv2d(
            input_size[0],
            kernel_count,
            kernel_size=init_kernel_size,
            stride=2,
            padding=init_kernel_size // 2,
            bias=False,
        )
        self.conv_imag.weight = torch.nn.Parameter(weights_imag, requires_grad=False)

        self.post_encoder = nn.Sequential(
            nn.BatchNorm2d(kernel_count),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                kernel_count,
                out_channels=input_size[0],
                kernel_size=init_kernel_size,
                stride=4,
                padding=init_kernel_size // 2,
                output_padding=3,
                bias=False,
            ),
            # nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(input_size[0]),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.conv_real(x) ** 2 + self.conv_imag(x) ** 2
        if self.use_abs:
            x = torch.sqrt(x)
        x = self.post_encoder(x)
        return x

    def decode(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.decoder(x)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.encode(x)
        x = self.decode(x)
        return x


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a random grayscale image (replace this with your actual image data)
    # image = np.random.rand(256, 256)

    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from load_dataset import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # construct the model
    model = V11V(input_size=(1, 448, 448), init_kernel_size=11, directions=7)
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
    columns = 7+1 # directions+1
    rows = model.conv_real.weight.shape[0] // columns

    fig, ax = plt.subplots(rows, columns, figsize=(20, 8))
    plt.ion()

    for row in range(rows):
        for column in range(columns):
            filter_weights = model.conv_real.weight[row * columns + column][0]
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
