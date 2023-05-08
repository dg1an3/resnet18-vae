import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_meshgrid(sz=7):
    return np.meshgrid(
        np.linspace(-(sz // 2), sz // 2, sz),
        np.linspace(-(sz // 2), sz // 2, sz),
    )


def complex_exp(xs, ys, freq, angle_rad):
    return np.exp(freq * (xs * np.sin(angle_rad) + ys * np.cos(angle_rad)) * 1.0j)


def gauss(xs, ys, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(
        -(xs * xs + ys * ys) / (2.0 * sigma * sigma)
    )


def gabor(xs, ys, freq, angle_rad, sigma=None):
    return complex_exp(xs, ys, freq, angle_rad) * gauss(
        xs, ys, sigma if sigma else 2.0 / freq
    )


def make_gabor_bank(xs, ys, directions=3, freqs=[2.0, 1.0]):
    # type: (...) -> List[np.ndarray]
    """ """
    for freq in freqs:
        for n in range(directions):
            angle = n * np.pi / np.float32(directions)
            kernel = gabor(xs, ys, freq, angle)
            yield kernel
        yield gauss(xs, ys, 2.0 / freq)


def kernels2weights(kernels, in_channels=1, dtype=torch.float32):
    kernels = np.repeat(kernels, in_channels, axis=0)
    kernels = np.expand_dims(kernels, axis=1)
    kernels = np.repeat(kernels, in_channels, axis=1)    
    return torch.tensor(kernels, dtype=dtype)


def make_oriented_map(
    inplanes: int = 3,
    kernel_size: int = 7,
    directions: int = 5,
    stride: int = 1,
    dstack_phases: bool = False,
):
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
    kernels_complex = (list)(
        make_gabor_bank(xs, ys, directions=directions, freqs=freqs)
    )

    if dstack_phases:
        conv = nn.Conv2d(
            inplanes,
            len(kernels_complex) * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=False,
        )

        stacked_weights = np.concatenate(
            (np.real(kernels_complex), np.imag(kernels_complex)), axis=0
        )
        print(f"stacked_weights.shape = {stacked_weights.shape}")

        conv.weight = torch.nn.Parameter(kernels2weights(stacked_weights, inplanes))

        raise("Not Implemented")
        return len(kernels_complex) * 2, conv
    else:
        #
        kernels_real, kernels_imag = np.real(kernels_complex), np.imag(kernels_complex)
        weights_real, weights_imag = (
            kernels2weights(kernels_real, inplanes),
            kernels2weights(kernels_imag, inplanes),
        )
        print(f"make_oriented_map: weights_real.shape = {weights_real.shape}")

        return len(kernels_complex), weights_real, weights_imag