from typing import Generator, Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_meshgrid(sz: int) -> List[np.ndarray]:
    """creates a 2d sz x sz grid of grid indices

    Args:
        sz (int): both grids will be sz x sz.

    Returns:
        List[np.ndarray]: a two-item list with the x and y value grids
    """
    return np.meshgrid(
        np.linspace(-(sz // 2), sz // 2, sz),
        np.linspace(-(sz // 2), sz // 2, sz),
    )


def complex_exp(
    xs: np.ndarray, ys: np.ndarray, freq: float, angle_rad: float
) -> np.ndarray[np.complex128]:
    """create a complex exponential kernel

    Args:
        xs (np.ndarray): x grid from make_meshgrid
        ys (np.ndarray): y grid from make_meshgrid
        freq (float): frequency of the exponential
        angle_rad (float): angle of k-vector, in radians

    Returns:
        np.ndarray[complex128]: complex-valued 2d array
    """
    return np.exp(freq * (xs * np.sin(angle_rad) + ys * np.cos(angle_rad)) * 1.0j)


def gauss(xs: np.ndarray, ys: np.ndarray, sigma: float) -> np.ndarray[np.float64]:
    """create a gauss kernel

    Args:
        xs (np.ndarray): x grid from make_meshgrid
        ys (np.ndarray): y grid from make_meshgrid
        sigma (float): sigma of the gaussian

    Returns:
        np.ndarray[float]: real-valued 2d array
    """
    return (1 / (2 * np.pi * sigma**2)) * np.exp(
        -(xs * xs + ys * ys) / (2.0 * sigma * sigma)
    )


def gabor(
    xs: np.ndarray, ys: np.ndarray, freq, angle_rad, sigma=None
) -> np.ndarray[np.complex128]:
    """create a gabor kernel

    Args:
        xs (np.ndarray): x grid from make_meshgrid
        ys (np.ndarray): y grid from make_meshgrid
        freq (float): frequency of the exponential
        angle_rad (float): angle of k-vector, in radians
        sigma (float): sigma of the gaussian

    Returns:
        np.ndarray[complex128]: complex-valued 2d array
    """
    return complex_exp(xs, ys, freq, angle_rad) * gauss(
        xs, ys, sigma if sigma else 2.0 / freq
    )


def make_gabor_bank(
    xs: np.ndarray, ys: np.ndarray, directions: int, freqs: List[float]
) -> Tuple[List[float], List[np.ndarray]]:
    """make a gabor bank with given directions and frequencies

    Args:
        xs (np.ndarray): x grid from make_meshgrid
        ys (np.ndarray): y grid from make_meshgrid
        directions (int): count of equally-spaced directions
        freqs (List[float]): list of frequencies to be generated

    Returns:
        Tuple[List[float], List[np.ndarray]]: pair of lists for frequencies and kernels
    """
    freq_per_kernel, kernels_complex = [], []

    for freq in freqs:
        freq_per_kernel.append(freq)

        kernel = gauss(xs, ys, 2.0 / freq)
        kernels_complex.append(kernel)

        for n in range(directions):
            freq_per_kernel.append(freq)

            angle = n * np.pi / np.float32(directions)
            kernel = gabor(xs, ys, freq, angle)
            kernels_complex.append(kernel)

    return freq_per_kernel, kernels_complex


def kernels2weights(
    kernels: np.ndarray, in_channels: int = 1, dtype=torch.float32
) -> torch.Tensor:
    """_summary_

    Args:
        kernels (np.ndarray): _description_
        in_channels (int, optional): _description_. Defaults to 1.
        dtype (_type_, optional): _description_. Defaults to torch.float32.

    Returns:
        torch.Tensor: _description_
    """
    # kernels = np.repeat(kernels, in_channels, axis=0)
    kernels = np.expand_dims(kernels, axis=1)
    kernels = np.repeat(kernels, in_channels, axis=1)
    return torch.tensor(kernels, dtype=dtype)


def make_oriented_map(
    in_channels: int = 3, kernel_size: int = 7, directions: int = 5
) -> Tuple[List[float], torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        kernel_size (int, optional): _description_. Defaults to 7.
        directions (int, optional): _description_. Defaults to 9.

    Returns:
        tuple: (in planes, real conv filter, imaginary conv filter)
    """
    xs, ys = make_meshgrid(sz=kernel_size)
    phi = (5**0.5 + 1) / 2  # golden ratio
    freqs = [phi**n for n in range(2, -3, -1)]
    freq_per_kernel, kernels_complex = make_gabor_bank(
        xs, ys, directions=directions, freqs=freqs
    )

    kernels_real, kernels_imag = np.real(kernels_complex), np.imag(kernels_complex)
    weights_real, weights_imag = (
        kernels2weights(kernels_real, in_channels),
        kernels2weights(kernels_imag, in_channels),
    )
    print(f"make_oriented_map: weights_real.shape = {weights_real.shape}")

    return freq_per_kernel, weights_real, weights_imag


def make_oriented_map_stack_phases(
    in_channels: int = 3, kernel_size: int = 7, directions: int = 5
) -> Tuple[List[float], torch.Tensor]:
    """_summary_

    Args:
        in_channels (int, optional): _description_. Defaults to 3.
        kernel_size (int, optional): _description_. Defaults to 7.
        directions (int, optional): _description_. Defaults to 5.

    Returns:
        Tuple[int, torch.Tensor]: _description_
    """
    freq_per_kernel, weights_real, weights_imag = make_oriented_map(
        in_channels, kernel_size, directions
    )

    stacked_freq_per_kernel = freq_per_kernel + freq_per_kernel
    stacked_kernels = torch.concatenate((weights_real, weights_imag), dim=0)
    return stacked_freq_per_kernel, stacked_kernels
