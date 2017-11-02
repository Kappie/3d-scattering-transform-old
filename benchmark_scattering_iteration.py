import numpy as np
from numba import vectorize, cuda, jit

from my_utils import extract_scattering_coefficients, extract_scattering_coefficients_cpu, time_me


if __name__ == '__main__':
    x = z = 128
    y = 256
    X = np.random.rand(x, y, z).astype(np.complex64)
    filter_fourier = np.random.rand(x, y, z).astype(np.complex64)
    downsampling_resolution = 1

    # compilation run
    time_me(extract_scattering_coefficients, X, filter_fourier, downsampling_resolution)

    # comparison runs
    result_gpu = extract_scattering_coefficients(X, filter_fourier, downsampling_resolution)
    result_cpu = extract_scattering_coefficients_cpu(X, filter_fourier, downsampling_resolution)

    print(np.average(result_gpu - result_cpu))

    print("gpu: ")
    time_me(extract_scattering_coefficients, X, filter_fourier, downsampling_resolution)
    print("cpu: ")
    time_me(extract_scattering_coefficients_cpu, X, filter_fourier, downsampling_resolution)
