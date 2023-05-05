from pathlib import Path
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import cv2
import numpy as np

# TODO: deprecate this in favor of Dataset-derived classes
# TODO: get from the environment
BASE_DATA_PATH = Path("D:\\") / "data"


def load_dataset(dataset_name="cxr8", input_size=(1, 448, 448), clahe_tile_size=8):
    """_summary_

    Args:
        dataset_name (str, optional): _description_. Defaults to "cxr8".
        input_size (tuple, optional): _description_. Defaults to (1, 448, 448).
        clahe_tile_size (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    if dataset_name == "cxr8":
        clahe = cv2.createCLAHE(
            clipLimit=4, tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
        cxr8_data_path = BASE_DATA_PATH / "cxr8" / "filtered"
        train_dataset = datasets.ImageFolder(
            cxr8_data_path,
            transform=transforms.Compose(
                [
                    transforms.Resize((input_size[1], input_size[2])),
                    transforms.Grayscale(),
                    transforms.Lambda(np.array),
                    transforms.Lambda(clahe.apply),
                    transforms.ToTensor(),
                ]
            ),
        )
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )
    elif dataset_name == "imagenet":
        imagenet_data_path = (
            BASE_DATA_PATH
            / "imagenet-object-localization-challenge"
            / "ILSVRC"
            / "Data"
            / "CLS-LOC"
        )
        train_dataset = datasets.ImageFolder(
            imagenet_data_path,
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )

    return train_dataset
