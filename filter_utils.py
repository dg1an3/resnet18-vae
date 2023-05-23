# -*- coding: utf-8 -*-
"""filter_utils.py implements a gabor pyramid for pytorch.

Notice the comment above the docstring specifying the encoding.
Docstrings do appear in the bytecode, so you can access this through
the ``__doc__`` attribute. This is also what you'll see if you call
help() on a module or any other Python object.
"""

from typing import Tuple, List, Union
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

    >>> xs,ys = make_meshgrid(sz=3)
    >>> xs
    [[-1,0,1],[-1,0,1],[-1,0,1]]
    >>> xs.shape
    (3, 3)
    >>> ys
    [[-1,-1,-1],[0,0,0],[1,1,1]]
    >>> ys.shape
    (3, 3)
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

    >>> xs,ys = mesh_grid(sz=3)
    >>> complex_exp(xs, ys, freq=1.0, angle_rad=0.13)
    [[-1-1j,0,1],[-1,0,1],[-1,0,1]]
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

    >>> xs,ys = mesh_grid(sz=3)
    >>> gauss(xs, ys, sigma=0.13)
    [[0.0]]
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

    >>> xs,ys = mesh_grid(sz=3)
    >>> gauss(xs, ys, sigma=0.13)
    [[0.0]]
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

    >>> xs,ys = mesh_grid(sz=3)
    >>> gauss(xs, ys, sigma=0.13)
    [[0.0]]        
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
        kernels (np.ndarray): array of kernels of out_channels x sz x sz
        in_channels (int, optional): number of in_channels for the resulting tensor. Defaults to 1.
        dtype (_type_, optional): type to be used for the kernels. Defaults to torch.float32.

    Returns:
        torch.Tensor: the resulting in_channels x out_channels x sz x sz tensor
    """
    # kernels = np.repeat(kernels, in_channels, axis=0)
    kernels = np.expand_dims(kernels, axis=1)
    kernels = np.repeat(kernels, in_channels, axis=1)
    return torch.tensor(kernels, dtype=dtype)


def make_oriented_map(
    in_channels: int = 3,
    kernel_size: int = 7,
    directions: int = 5,
    frequencies: Union[None, List[float]] = None,
) -> Tuple[List[float], torch.Tensor, torch.Tensor]:
    """constructs an oriented map with both real and imaginary components.

    Args:
        in_channels (int, optional): number of input channels to support. Defaults to 3.
        kernel_size (int, optional): size of each kernel is uniform in width and height, size should be odd. Defaults to 7.
        directions (int, optional): number of directions per spatial frequency. Defaults to 9.
        frequencies (Union[None, List[float]], optional): _description_. Defaults to None.

    Returns:
        tuple: (in planes, real conv filter, imaginary conv filter)
    """
    xs, ys = make_meshgrid(sz=kernel_size)

    if frequencies is None:
        # populate with standard golden ratio frequencies
        phi = (5**0.5 + 1) / 2  # golden ratio
        frequencies = [phi**n for n in range(1, -4, -1)]

    # construct the gabor bank (which is a complex-valued tensor)
    freq_per_kernel, kernels_complex = make_gabor_bank(
        xs, ys, directions=directions, freqs=frequencies
    )

    # extract real and imaginary components
    kernels_real, kernels_imag = np.real(kernels_complex), np.imag(kernels_complex)

    # turn in to weights (single tensor for in_channels)
    weights_real, weights_imag = (
        kernels2weights(kernels_real, in_channels),
        kernels2weights(kernels_imag, in_channels),
    )
    print(f"make_oriented_map: weights_real.shape = {weights_real.shape}")

    return freq_per_kernel, weights_real, weights_imag


def make_oriented_map_stack_phases(
    # in_channels: int = 3,
    # kernel_size: int = 7,
    # directions: int = 5,
    # frequencies: Union[None, List[float]] = None,
    **kwargs,
) -> Tuple[List[float], torch.Tensor]:
    """stacks together the real and imaginary phases of the oriented map

    Args:
        in_channels (int, optional): number of input channels to support. Defaults to 3.
        kernel_size (int, optional): size of each kernel is uniform in width and height, size should be odd. Defaults to 7.
        directions (int, optional): number of directions per spatial frequency. Defaults to 9.
        frequencies (Union[None, List[float]], optional): _description_. Defaults to None.

    Returns:
        Tuple[int, torch.Tensor]: _description_
    """
    freq_per_kernel, weights_real, weights_imag = make_oriented_map(
        # in_channels=in_channels, kernel_size=kernel_size, directions=directions, frequencies=frequencies,
        **kwargs
    )

    stacked_freq_per_kernel = freq_per_kernel + freq_per_kernel
    stacked_kernels = torch.concatenate((weights_real, weights_imag), dim=0)
    return stacked_freq_per_kernel, stacked_kernels


#######################################################################################
#######################################################################################
#     ###     ###     ###     ###     ###     ###     ###     ###
#     ###     ###     ###     ###     ###     ###     ###     ###
#     ###     ###     ###     ###     ###     ###     ###     ###
#######################################################################################
#######################################################################################

import unittest


class TestFilterUtils(unittest.TestCase):
    def test_make_meshgrid(self):
        xs, ys = make_meshgrid(sz=7)
        self.assertEqual(xs.shape, (7, 7))
        self.assertEqual(xs.shape, ys.shape)

    def test_make_gabor_bank(self):
        xs, ys = make_meshgrid(sz=7)
        directions = 3
        freqs = [2.0, 1.0, 0.5]
        freq_per_kernel, kernels_complex = make_gabor_bank(
            xs, ys, directions=directions, freqs=freqs
        )
        self.assertEqual(len(kernels_complex), (len(freqs) + 1) * directions)
        for freq, kernel in zip(freq_per_kernel, kernels_complex):
            self.assertIsInstance(freq, float)
            self.assertIsInstance(kernel, np.ndarray)
            self.assertIn(kernel.dtype, [np.float64, np.complex128])
            self.assertEqual(xs.shape, kernel.shape)

    def test_kernels2weight(self):
        xs, ys = make_meshgrid(sz=7)
        directions = 3
        freqs = [2.0, 1.0, 0.5]
        _, kernels_complex = make_gabor_bank(xs, ys, directions=directions, freqs=freqs)
        kernels_real, kernels_imag = np.real(kernels_complex), np.imag(kernels_complex)
        self.assertIsInstance(kernels_real, np.ndarray)
        self.assertIsInstance(kernels_imag, np.ndarray)

        in_channels = 17
        kernels_real_1 = kernels2weights(kernels_real, in_channels)
        # print(kernels_real.shape)
        print(kernels_real_1.shape)
        self.assertEqual(kernels_real_1.shape[0], len(kernels_complex))
        self.assertEqual(kernels_real_1.shape[1], in_channels)
        self.assertEqual(kernels_real_1.shape[2], 7)
        self.assertEqual(kernels_real_1.shape[3], 7)

        kernels_imag_1 = kernels2weights(kernels_imag, in_channels)
        self.assertEqual(kernels_imag_1.shape[0], len(kernels_complex))
        self.assertEqual(kernels_imag_1.shape[1], in_channels)
        self.assertEqual(kernels_imag_1.shape[2], 7)
        self.assertEqual(kernels_imag_1.shape[3], 7)

    def test_make_oriented_map(self):
        xs, ys = make_meshgrid(sz=7)
        self.assertEqual(xs.shape, (7, 7))
        self.assertEqual(xs.shape, ys.shape)

    def test_make_oriented_map_stack_phases(self):
        xs, ys = make_meshgrid(sz=7)
        self.assertEqual(xs.shape, (7, 7))
        self.assertEqual(xs.shape, ys.shape)


if __name__ == "__main__":
    unittest.main()
