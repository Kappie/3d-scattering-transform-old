import numpy as np
import numba
import math
import matplotlib.pyplot as plt

from fibonacci_spiral import rotation_matrices_fibonacci_spiral_unit_x
from plot_slices import plot3d
from my_utils import crop_freq_3d


def filter_bank(dimensions, js, J, n_points_fourier_sphere, sigma, xi):
    """
    js: length scales for filters. Filters will be dilated by 2**j for j in js.
    J: length scale used for averaging over scattered signals. (coefficients will be approximately translationally
    invariant over 2**J pixels.)
    n_points_fourier_sphere: number of rotations from rotation group on unit sphere.

    We store all signals in fourier space.
    TODO: normalize Morlets.
    """
    filters = {}
    filters['psi'] = []

    rotation_matrices = rotation_matrices_fibonacci_spiral_unit_x(n_points_fourier_sphere)

    for j, r in product(js, rotation_matrices):
        psi = {'j': j, 'r': r}
        psi_signal_fourier = morlet_fourier_3d(dimensions, j, r, xi, sigma)
        # When j_1 < j_2 < ... < j_n, we need j_2, ..., j_n downsampled at j_1, j_3, ..., j_n downsampled at j_2, etc.
        # resolution 0 is just the signal itself. See below header "Fast scattering computation" in Bruna (2013).
        for resolution in range(j + 1):
            psi_signal_fourier_res = crop_freq_3d(psi_signal, resolution)
            psi[resolution] = psi_signal_fourier_res

        filters['psi'].append(psi)

    filters['phi'] = {}
    filters['phi']['j'] = J
    phi_signal_fourier = gaussian_filter_3d(dimensions, J, sigma)
    phi_signal_fourier = normalize_fourier(phi_signal_fourier)
    # We need the phi signal downsampled at all length scales j.
    for resolution in js:
        phi_signal_fourier_res = crop_freq_3d(phi_signal_fourier, resolution)
        filters['phi'][resolution] = phi_signal_fourier_res

    return filters

@numba.jit
def morlet_fourier_3d(dimensions, j, r, xi, sigma, a=2.0):
    """
    Assumes dimensions are powers of two.
    r: 3x3 rotation matrix.
    xi: [xi, 0, 0] by convention.
    """
    width, height, depth = dimensions
    result = np.empty((width, height, depth))

    scale_factor = a**j
    normalisation = a**(-3*j)
    kappa_sigma = gauss_3d(-xi[0], -xi[1], -xi[2], sigma) / gauss_3d(0, 0, 0, sigma)
    for k in range(-width//2, width//2):
        for l in range(-height//2, height//2):
            for m in range(-depth//2, depth//2):
                # Rotate and scale.
                k_prime = (r[0, 0]*k + r[0, 1]*l + r[0, 2]*m) * scale_factor
                l_prime = (r[1, 0]*k + r[1, 1]*l + r[1, 2]*m) * scale_factor
                m_prime = (r[2, 0]*k + r[2, 1]*l + r[2, 2]*m) * scale_factor
                result[k, l, m] = normalisation * (
                    gauss_3d(k_prime-xi[0], l_prime-xi[1], m_prime-xi[2], sigma) -
                    kappa_sigma*gauss_3d(k_prime, l_prime, m_prime, sigma) )
    return result


@numba.jit
def gaussian_filter_3d(dimensions, j, sigma, a=2.0):
    width, height, depth = dimensions
    result = np.empty((width, height, depth))

    scale_factor = a**j
    for k in range(-width//2, width//2):
        for l in range(-height//2, height//2):
            for m in range(-depth//2, depth//2):
                result[k, l, m] = gauss_3d(k, l, m, sigma/scale_factor)
    return result


@numba.jit
def morlet_fourier_1d(N, j, xi, sigma, a=2.0):
    """
    Assumes signal length N = 2^n.
    """
    result = np.empty(N)
    kappa_sigma = gauss_1d(-xi, sigma) / gauss_1d(0, sigma)
    normalisation = a**(-j)

    for omega in range(-N//2, N//2):
        result[omega] = normalisation * ( gauss_1d(a**j * omega - xi, sigma) - kappa_sigma*gauss_1d(a**j * omega, sigma) )
    return result


@numba.jit
def gauss_3d(x, y, z, sigma):
    return math.exp(-(x*x + y*y + z*z) / (2*sigma*sigma))


@numba.jit
def gauss_1d(x, sigma):
    return math.exp(-x*x / (2*sigma*sigma))


def normalize_fourier(signal_fourier):
    """
    Normalising in Fourier domain means making sure the zeroth frequency component
    is equal to 1.
    """
    return signal_fourier / signal_fourier[0, 0, 0]


def plot(x):
    plt.plot(x)
    plt.show()


if __name__ == '__main__':
    # N = 64
    # xi_radians = 4*np.pi/5
    # xi = N * xi_radians/(2*np.pi)
    # sigma_spatial = 0.6
    # sigma_fourier = 1 / sigma_spatial
    #
    # for j in range(4):
    #     result = morlet_fourier_1d(N, j, xi, sigma_fourier)
    #     print(result[0])
    #     plot(result)

    dimensions = np.array([64, 64, 64])
    xi_radians = 4*np.pi/5
    xi = np.ceil(dimensions[0] * xi_radians/(2*np.pi))
    xi = np.array([xi, 0, 0])
    sigma_spatial = 0.2
    sigma_fourier = 1 / sigma_spatial
    n_points_fourier_sphere = 4
    rotation_matrices = rotation_matrices_fibonacci_spiral_unit_x(n_points_fourier_sphere)

    # for j in range(3):
    #     # for r in rotation_matrices:
    #     r = np.eye(3)
    #     result = morlet_fourier_3d(dimensions, j, r, xi, sigma_fourier)
    #     print(result[0, 0, 0])
    #     maximum_pos = np.unravel_index(np.argmax(result), result.shape)
    #     print(maximum_pos)
    #     plot3d(result)
    # for j in range(3):
    #     result = gaussian_filter_3d(dimensions, j, sigma_fourier)
    #     result = normalize_fourier(result)
    #     plot3d(result)
