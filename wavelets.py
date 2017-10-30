import numpy as np
import tensorflow as tf
import scipy.signal
import scipy.fftpack as fft
from transforms3d.euler import euler2mat
from cube_show_slider import cube_show_slider
from itertools import product
import numba
import time


# def mother_gabor(u, xi, sigma):
#     return np.exp(-np.dot(u, u)/(2*sigma**2) + 1j*np.dot(xi, u))
#
#
# def gabor_filter_old(width, height, depth, j, alpha, beta, gamma, xi=np.array([3*np.pi/4, 0, 0]), sigma=1, a=2):
#     """
#     xi: modulation of periodic part of wavelet
#     sigma: standard deviation of gaussian window
#     a: scale parameter
#     """
#     rotation_matrix = euler2mat(alpha, beta, gamma, 'sxyz')
#
#     gab_filter = np.zeros([width, height, depth], dtype=np.complex)
#     for x in range(width):
#         for y in range(height):
#             for z in range(depth):
#                 centered_x, centered_y, centered_z = center(x, width), center(y, height), center(z, depth)
#                 rotated_u = np.dot(rotation_matrix, np.array([centered_x, centered_y, centered_z]))
#                 gab_filter[x, y, z] = (a**-j)*(1/sigma)*mother_gabor(rotated_u*a**-j, xi, sigma)
#
#     return gab_filter
#
#
#
#
# def gabor_filters(J, alphas, betas, gammas):
#     return [ [ [ [ gabor_filter(j, alpha, beta, gamma) for gamma in gammas ] for beta in betas ] for alpha in alphas ] for j in range(J) ]
#
#
# def center(index, list_length):
#     return int(np.floor(index - (list_length-1)/2))


# def crop_freq_3d_old(x, res):
#     """
#     Crop highest (1 - 2^-res) part of a fourier spectrum.
#     (So for res = 1, cut highest half of the spectrum, res = 2 cut highest 3/4, etc.)
#     Corresponds to a spatial downsampling of the image by a factor (res + 1).
#     """
#     M, N, O = x.shape[0], x.shape[1], x.shape[2]
#     # Dimensions after cropping
#     A, B, C = [int(dim // 2**res) for dim in [M, N, O]]
#
#     crop = np.zeros((A, B, C), np.complex64)
#     mask = np.ones(x.shape, np.float32)
#
#     len_x, len_y, len_z = [int(dim * (1 - 2 ** (-res))) for dim in [M, N, O]]
#     start_x, start_y, start_z = [int(dim * 2 ** (-res - 1)) for dim in [M, N, O]]
#     # Crop highest frequencies
#     mask[start_x:start_x + len_x, :, :] = 0
#     mask[:, start_y:start_y + len_y, :] = 0
#     mask[:, :, start_z:start_z + len_z] = 0
#     x = np.multiply(x,mask)
#
#     # Rescale spectrum? Or rather, rearrange the coefficients such that
#     # the remaining ones fit into a smaller array, making use of the symmetry
#     # of fourier spectrum of a real signal (e.g. an image).
#     for a in range(A):
#         for b in range(B):
#             for c in range(C):
#                 for i in range(int(2 ** res)):
#                     for j in range(int(2 ** res)):
#                         for k in range(int(2 ** res)):
#                             crop[a, b, c] += x[a + i*A, b + j*B, c + k*C]
#
#     return crop

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


def crop_freq_3d(x, res):
    """
    Crop highest (1 - 2^-res) part of a fourier spectrum.
    (So for res = 1, cut highest half of the spectrum, res = 2 cut highest 3/4, etc.)
    This comes down to only taking the dim/(2**(res+1)) elements at the front and end of each dimension of the original array.
    In 2D, for res = 1 and a 4x4 input, you would get (taking only the single element at the front and back of each dimension)
    [[a00 a03], [a30, a33]]
    Corresponds to a spatial downsampling of the image by a factor (res + 1).
    Expects dimensions of array to be powers of 2.
    """
    if res == 0:
        return x

    M, N, O = x.shape[0], x.shape[1], x.shape[2]
    end_x, end_y, end_z = [int(dim * 2 ** (-res - 1)) for dim in [M, N, O]]
    indices_x, indices_y, indices_z = [ list(range(end_index)) + list(range(-end_index, 0)) for end_index in [end_x, end_y, end_z] ]
    indices = np.ix_(indices_x, indices_y, indices_z)
    return x[indices]


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
        psi_signal_fourier = fft.fftn(psi_signal)
        # When j_1 < j_2 < ... < j_n, we need j_2, ..., j_n downsampled at j_1, j_3, ..., j_n downsampled at j_2, etc.
        # resolution 0 is just the signal itself. See below header "Fast scattering computation" in Bruna (2013).
        for resolution in range(j + 1):
            psi_signal_fourier_res = crop_freq_3d(psi_signal_fourier, resolution)
            psi[resolution] = normalize(psi_signal_fourier_res, width, height, depth, j)

        filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = gaussian_filter(width, height, depth, J)
    phi_signal_fourier = fft.fftn(phi_signal)
    filters['phi']['j'] = J
    # We need the phi signal downsampled at all length scales j.
    for resolution in js:
        phi_signal_fourier_res = crop_freq_3d(phi_signal_fourier, resolution)
        filters['phi'][resolution] = normalize(phi_signal_fourier_res, width, height, depth, J)

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
