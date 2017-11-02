import numpy as np
from numba import vectorize, cuda, jit

from my_utils import extract_scattering_coefficients


if __name__ == '__main__':
    x = y = z = 128
    X_fourier = np.random.rand(x, y, z).astype(np.complex64)
    filter_fourier = np.random.rand(x, y, z).astype(np.complex64)
    downsampling_resolution = 1

    extract_scattering_coefficients(X_fourier, filter_fourier, downsampling_resolution)
