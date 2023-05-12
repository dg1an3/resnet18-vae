import unittest

from filter_utils import *


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
