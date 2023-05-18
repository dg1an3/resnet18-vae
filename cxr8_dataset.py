import logging
import torch
from torchvision import transforms
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
import SimpleITK as sitk

def match_histograms(fixed, moving):
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)

    matcher = sitk.HistogramMatchingImageFilter()
    if fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    moving = sitk.GetArrayFromImage(moving)

    return moving


def get_clahe_transforms(clip_limit=4, clahe_tile_size=8, input_size=448):
    """_summary_

    Args:
        clahe_tile_size (int, optional): _description_. Defaults to 8.
        input_size (tuple, optional): _description_. Defaults to (448, 448).

    Returns:
        _type_: _description_
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size)
    )

    return transforms.Compose(
        [
            # transforms.Grayscale(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,),(10.0,)),
            # transforms.Lambda(lambda x: x*255.0),
            # transforms.ToPILImage(),
            # transforms.Resize((input_size, input_size)),

            # transforms.Lambda(np.array),
            # transforms.Lambda(clahe.apply),
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

        clip_limit=4
        clahe_tile_size=8
        input_size=448
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size)
    )


    def read_img_file(self, img_name):
        img_name = self.root_path / "images" / img_name
        img_name = str(img_name)
        # print(f"img_name {img_name}")

        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (448,448))
        # image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX)
        image = self.clahe.apply(image)
        image_min, image_max, image_avg, image_std = np.min(image), np.max(image), np.average(image), np.std(image)
        logging.debug(f"image min/max/avg = {image_min}, {image_max}, {image_avg:.4f}, {image_std:.4f}")
        #image = (image + (128.0 - image_avg))/256.0
        image = 0.5 + (image - image_avg)/(4.0*image_std)
        image = image.astype(np.float32)
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
            raise ("unknown type")

        finding_labels = self.data_entry_df["Finding Labels"].iloc[idx]
        if isinstance(finding_labels, pd.Series):
            finding_labels = finding_labels.str  # .split("|")
        elif isinstance(finding_labels, str):
            finding_labels = finding_labels  # .split("|")
        else:
            raise ("unknown type")

        if self.transform:
            images = self.transform(images)

        samples = {"image": images, "labels": finding_labels}

        return samples
