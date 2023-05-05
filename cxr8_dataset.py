import torch
from torchvision import transforms
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2


def get_clahe_transforms(clahe_tile_size=8, input_size=448):
    """_summary_

    Args:
        clahe_tile_size (int, optional): _description_. Defaults to 8.
        input_size (tuple, optional): _description_. Defaults to (448, 448).

    Returns:
        _type_: _description_
    """
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=clahe_tile_size)

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(),
            transforms.Lambda(np.array),
            transforms.Lambda(clahe.apply),
            transforms.ToTensor(),
        ]
    )


class Cxr8Dataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, root_path, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_path = root_path if root_path is Path else Path(root_path)

        csv_filename = self.root_path / "Data_Entry_2017_v2020.csv"
        self.data_entry_df = pd.read_csv(csv_filename)

        self.transform = transform

    def read_img_file(self, img_name):
        img_name = self.root_path / "images" / img_name
        img_name = str(img_name)
        # print(f"img_name {img_name}")

        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        return image

    def __len__(self):
        return len(self.data_entry_df)

    def __str__(self):
        return f"{type(self)}: Dataset at {self.root_path} with {len(self)} items."

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_index = self.data_entry_df["Image Index"].iloc[idx]
        if isinstance(image_index, pd.Series):
            images = image_index.apply(self.read_img_file)
        elif isinstance(image_index, str):
            images = self.read_img_file(image_index)
        else:
            raise("unknown type")
        
        finding_labels = self.data_entry_df["Finding Labels"].iloc[idx]
        if isinstance(finding_labels, pd.Series):
            finding_labels = finding_labels.str.split("|")
        elif isinstance(finding_labels, str):
            finding_labels = finding_labels.split("|")
        else:
            raise("unknown type")

        sample = {"image": images, "labels": finding_labels}

        if self.transform:
            sample = self.transform(sample)

        return sample