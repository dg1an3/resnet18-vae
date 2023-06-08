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


class Cxr8Dataset(Dataset):
    """class that represents the CXR8 chest x-ray dataset, with some pre-processing"""

    def __init__(self, root_path: str, transform=None, sz: int = 1024):
        """_summary_

        Args:
        Args:
            root_path (string): Directory with all the images.
            transform (_type_, optional):  Optional transform to be applied
                on a sample. Defaults to None.
            sz (int, optional): size of input image. Defaults to 1024.
            Actual input channels will be original, clahe_1, clahe_2, and
            oriented pyramids of the same, reduced to fixed channels
            TODO: include first oriented map as part of pre processing
        """
        self.root_path = root_path if root_path is Path else Path(root_path)

        csv_filename = self.root_path / "Data_Entry_2017_v2020.csv"
        self.data_entry_df = pd.read_csv(csv_filename)

        self.transform = transform
        self.input_size = (sz, sz)

        clip_limit = 4
        self.clahe_16 = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
        self.clahe_8 = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

        self.clahe_4 = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))

    def read_img_file(self, img_name):
        img_name = self.root_path / "images" / img_name
        img_name = str(img_name)
        # print(f"img_name {img_name}")

        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.input_size)
        image_avg, image_std = (
            np.average(image),
            np.std(image),
        )
        image = 0.5 + (image - image_avg) / (3.0 * image_std)

        # image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_clahe_16 = self.apply_clahe(self.clahe_16, image)
        image_clahe_8 = self.apply_clahe(self.clahe_8, image)
        image_clahe_4 = self.apply_clahe(self.clahe_4, image)

        image_result = np.stack(
            [image, image_clahe_4, image_clahe_8, image_clahe_16], axis=-1
        )
        return image_result.astype(np.float32)

    def apply_clahe(self, clahe_filter, image):
        image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)
        image = clahe_filter.apply(image)
        image_min, image_max, image_avg, image_std = (
            np.min(image),
            np.max(image),
            np.average(image),
            np.std(image),
        )
        logging.debug(
            f"image min/max/avg = {image_min}, {image_max}, {image_avg:.4f}, {image_std:.4f}"
        )
        # normalize to 4*std
        image = 0.5 + (image - image_avg) / (3.0 * image_std)
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
