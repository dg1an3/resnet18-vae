from pathlib import Path
import cv2

import torch
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode
from torchvision.datasets import Flowers102


def normalize_tensor(t):
    assert len(t.shape) < 4 or (t.shape[0] == 1)
    t_std, t_avg = torch.std_mean(t)
    return 0.5 + (t - t_avg) / (3.0 * t_std)


class ConvertColor(object):
    def __init__(self, code=cv2.COLOR_BGR2YCR_CB):
        self.code = code

    def __call__(self, sample):
        print(f"sample before conversion = {sample.shape}")
        sample = sample.permute((1, 2, 0))
        sample = torch.tensor(cv2.cvtColor(sample.numpy(), self.code))
        sample = sample.permute((2, 0, 1))
        print(f"sample after conversion = {sample.shape}")
        return sample


class ClaheFilter(object):
    def __init__(self, clip_limit, tile_grid_sizes):
        self.filters = []
        for tile_grid_size in tile_grid_sizes:
            self.filters.append(
                cv2.createCLAHE(
                    clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
                )
            )

    def __call__(self, samples):
        channels = [samples[:, n, ...] for n in range(3)]

        # normalize image
        channels[0] = normalize_tensor(channels[0])
        channels[0] *= 255.0
        channels[0] = torch.clip(channels[0], 0.0, 255.0)
        channels[0] = channels[0].to(torch.uint8)

        for clahe_filter in self.filters:
            filtered_slices = []
            for s in range(samples.shape[0]):
                slice = samples[s, ...].numpy()
                filtered_slices.append(torch.tensor(clahe_filter.apply(slice)))
            filtered = normalize_tensor(filtered_slices)
            channels.append(filtered)

        result = torch.stack(channels, 1)
        result = result.to(torch.float32)
        return result


def get_oxflow_dataset(path=None, sz=512):
    if path is None:
        path = Path("c:") / "data" / "flowers"

    return Flowers102(
        path,
        # split="train",
        download=True,
        transform=Compose(
            [
                ToTensor(),
                ConvertColor(code=cv2.COLOR_BGR2YCR_CB),
                Resize((sz, sz), interpolation=InterpolationMode.BILINEAR),
                ClaheFilter(clip_limit=4, tile_grid_size=[8, 16, 32]),
            ]
        ),
    )


if __name__ == "__main__":
    flds = get_oxflow_dataset(path=None)
    print(flds)
