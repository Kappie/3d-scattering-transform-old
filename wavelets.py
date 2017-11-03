import numpy as np
import tensorflow as tf
import scipy.signal
import math
import cmath
import time
from transforms3d.euler import euler2mat
from cube_show_slider import cube_show_slider
from itertools import product
from numba import cuda, vectorize, jit
from gpu_vs_cpu_test import time_me

from my_utils import crop_freq_3d, fourier, get_blocks_and_threads, downsample


XI_DEFAULT = np.array([3*np.pi/4, 0, 0])
A_DEFAULT = 2
SIGMA_DEFAULT = 1.


def get_gabor_filter_gpu(width, height, depth, j, alpha, beta, gamma, xi=XI_DEFAULT, a=A_DEFAULT, sigma=SIGMA_DEFAULT):
    R = euler2mat(alpha, beta, gamma, 'sxyz')
    result = np.empty((width, height, depth), dtype=np.complex64)
    blockspergrid, threadsperblock = get_blocks_and_threads(width, height, depth)
    gabor_filter_gpu[blockspergrid, threadsperblock](width, height, depth, j, R, xi, a, sigma, result)
    return result


def get_gabor_filter(width, height, depth, j, alpha, beta, gamma, xi=XI_DEFAULT, a=A_DEFAULT, sigma=SIGMA_DEFAULT):
    R = euler2mat(alpha, beta, gamma, 'sxyz')
    result = np.empty((width, height, depth), dtype=np.complex64)
    gabor_filter_compiled(width, height, depth, j, R, xi, a, sigma, result)
    return result


def get_gaussian_filter(width, height, depth, j, a=A_DEFAULT, sigma=SIGMA_DEFAULT):
    result = np.empty((width, height, depth), dtype=np.float32)
    gaussian_filter_compiled(width, height, depth, j, a, sigma, result)
    return result


def get_gaussian_filter_gpu(width, height, depth, j, a=A_DEFAULT, sigma=SIGMA_DEFAULT):
    result = np.empty((width, height, depth), dtype=np.float32)
    blockspergrid, threadsperblock = get_blocks_and_threads(width, height, depth)
    gaussian_filter_gpu[blockspergrid, threadsperblock](width, height, depth, j, a, sigma, result)
    return result


@jit(['void(int64, int64, int64, int64, float64[:, :], float64[:], int64, float64, complex64[:, :, :])'], nopython=True)
def gabor_filter_compiled(width, height, depth, j, R, xi, a, sigma, result):
    scale_factor = 1/(sigma*a**j)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                x_prime = (R[0, 0]*x + R[0, 1]*y + R[0, 2]*z) * scale_factor
                y_prime = (R[1, 0]*x + R[1, 1]*y + R[1, 2]*z) * scale_factor
                z_prime = (R[2, 0]*x + R[2, 1]*y + R[2, 2]*z) * scale_factor
                result[x, y, z] = scale_factor * cmath.exp(-(x_prime**2 + y_prime**2 + z_prime**2)/2 + 1j*sigma*(x_prime*xi[0] + y_prime*xi[1] + z_prime*xi[2]))


@cuda.jit()
def gabor_filter_gpu(width, height, depth, j, R, xi, a, sigma, result):
    x, y, z = cuda.grid(3)
    scale_factor = 1/(sigma*a**j)
    if x < width and y < height and z < depth:
        x_prime = (R[0, 0]*x + R[0, 1]*y + R[0, 2]*z) * scale_factor
        y_prime = (R[1, 0]*x + R[1, 1]*y + R[1, 2]*z) * scale_factor
        z_prime = (R[2, 0]*x + R[2, 1]*y + R[2, 2]*z) * scale_factor
        result[x, y, z] = scale_factor * cmath.exp(-(x_prime**2 + y_prime**2 + z_prime**2)/2 + 1j*sigma*(x_prime*xi[0] + y_prime*xi[1] + z_prime*xi[2]))


@jit(['void(int64, int64, int64, int64, int64, float64, float32[:, :, :])'], nopython=True)
def gaussian_filter_compiled(width, height, depth, j, a, sigma, result):
    scale_factor = 1/(sigma*a**j)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                result[x, y, z] = scale_factor * math.exp(-(x**2 + y**2 + z**2)*scale_factor/2)


@cuda.jit()
def gaussian_filter_gpu(width, height, depth, j, a, sigma, result):
    # TODO: centering of arrays??
    x, y, z = cuda.grid(3)
    scale_factor = 1/(sigma*a**j)
    if x < width and y < height and z < depth:
        result[x, y, z] = scale_factor * math.exp(-(x**2 + y**2 + z**2)*scale_factor/2)


def filter_bank(width, height, depth, js, J, L):
    """
    js: length scales for filters. Filters will be dilated by 2**j for j in js.
    J: length scale used for averaging over scattered signals. (coefficients will be approximately translationally
    invariant over 2**J pixels.)
    L: number of angles for filters, spaced evenly in (0, pi).
    """
    filters = {}
    filters['psi'] = []

    alphas = betas = gammas = [(n/(L-1)) * np.pi for n in range(L)]

    for j, alpha, beta, gamma in product(js, alphas, betas, gammas):
        psi = {'j': j, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
        psi_signal = get_gabor_filter_gpu(width, height, depth, j, alpha, beta, gamma)
        psi_signal_fourier = fourier(psi_signal)
        # When j_1 < j_2 < ... < j_n, we need j_2, ..., j_n downsampled at j_1, j_3, ..., j_n downsampled at j_2, etc.
        # resolution 0 is just the signal itself. See below header "Fast scattering computation" in Bruna (2013).
        for resolution in range(j + 1):
            psi_signal_fourier_res = crop_freq_3d(psi_signal_fourier, resolution)
            psi[resolution] = psi_signal_fourier_res

        filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = get_gaussian_filter_gpu(width, height, depth, J)
    phi_signal_fourier = fourier(phi_signal)
    filters['phi']['j'] = J
    # We need the phi signal downsampled at all length scales j.
    for resolution in js:
        phi_signal_fourier_res = crop_freq_3d(phi_signal_fourier, resolution)
        filters['phi'][resolution] = phi_signal_fourier_res

    return filters


if __name__ == '__main__':
    y = 256
    x = z = 128
    js = [0, 1, 2, 3]
    J = 3
    L = 4

    start = time.time()
    filters = filter_bank(x, y, z, js, J, L)
    end = time.time()
    print(len(filters['psi']))
    print((end - start) / len(filters['psi']))
