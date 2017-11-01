import numpy as np
import tensorflow as tf
import scipy.signal
from transforms3d.euler import euler2mat
from cube_show_slider import cube_show_slider
from itertools import product
from numba import cuda, vectorize
from gpu_vs_cpu_test import time_me

from my_utils import crop_freq_3d, fourier


def gabor_filter(width, height, depth, j, alpha, beta, gamma, xi=np.array([3*np.pi/4, 0, 0]), a=2, sigma=1):
    """
    Outputs gabor filter of shape `dimensions`.
    """
    x = centered_array(width)
    y = centered_array(height)
    z = centered_array(depth)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    # Apply rotation matrix to coordinates and scale by a**j.
    R = euler2mat(alpha, beta, gamma, 'sxyz')
    xx_prime = (R[0, 0]*xx + R[0, 1]*yy + R[0, 2]*zz) / a**j
    yy_prime = (R[1, 0]*xx + R[1, 1]*yy + R[1, 2]*zz) / a**j
    zz_prime = (R[2, 0]*xx + R[2, 1]*yy + R[2, 2]*zz) / a**j
    # Apply gabor function and multiply by 1 / (a**j * sigma), as in the original
    # definition of scaled and rotated wavelet. (See e.g. Adel et al 2016.)
    gab_filter = (1/(a**j * sigma)) * np.exp(-(xx_prime**2 + yy_prime**2 + zz_prime**2)/(2*sigma**2) + 1j*(xx_prime*xi[0] + yy_prime*xi[1] + zz_prime*xi[2]))
    return gab_filter


def gaussian_filter(width, height, depth, J):
    return np.real( gabor_filter(width, height, depth, J, 0, 0, 0, xi=np.array([0, 0, 0])) )


def centered_array(size):
    """
    Returns [-size//2, ..., -2, -1, 0, 1, 2, ..., size//2]
    """
    return np.arange(size, dtype=np.float) - size // 2


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
        psi_signal = gabor_filter(width, height, depth, j, alpha, beta, gamma)
        psi_signal_fourier = fourier(psi_signal)
        # When j_1 < j_2 < ... < j_n, we need j_2, ..., j_n downsampled at j_1, j_3, ..., j_n downsampled at j_2, etc.
        # resolution 0 is just the signal itself. See below header "Fast scattering computation" in Bruna (2013).
        for resolution in range(j + 1):
            psi_signal_fourier_res = crop_freq_3d(psi_signal_fourier, resolution)
            psi[resolution] = psi_signal_fourier_res

        filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = gaussian_filter(width, height, depth, J)
    phi_signal_fourier = fourier(phi_signal)
    filters['phi']['j'] = J
    # We need the phi signal downsampled at all length scales j.
    for resolution in js:
        phi_signal_fourier_res = crop_freq_3d(phi_signal_fourier, resolution)
        filters['phi'][resolution] = phi_signal_fourier_res

    return filters


def normalize(signal, width, height, depth, j):
    return signal / (width * height * depth // 2**(2*j))


if __name__ == '__main__':
    y = 128
    x = z = 64
    js = [0, 1, 2]
    J = 3
    L = 3

    start = time.time()
    filters = filter_bank(x, y, z, js, J, L)
    end = time.time()
    print(end - start)
