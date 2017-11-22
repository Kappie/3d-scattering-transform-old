import numpy as np
import numba
import math
import matplotlib.pyplot as plt

from fibonacci_spiral import rotation_matrices_fibonacci_spiral_unit_x
from plot_slices import plot3d


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
    for k_idx, k in enumerate(range(-width//2, width//2)):
        for l_idx, l in enumerate(range(-height//2, height//2)):
            for m_idx, m in enumerate(range(-depth//2, depth//2)):
                # Rotate and scale.
                k_prime = (r[0, 0]*k + r[0, 1]*l + r[0, 2]*m) * scale_factor
                l_prime = (r[1, 0]*k + r[1, 1]*l + r[1, 2]*m) * scale_factor
                m_prime = (r[2, 0]*k + r[2, 1]*l + r[2, 2]*m) * scale_factor
                result[k_idx, l_idx, m_idx] = normalisation * (
                    gauss_3d(k_prime-xi[0], l_prime-xi[1], m_prime-xi[2], sigma) -
                    kappa_sigma*gauss_3d(k_prime, l_prime, m_prime, sigma) )
    return result


@numba.jit
def morlet_fourier_1d(N, j, xi, sigma, a=2.0):
    """
    Assumes signal length N = 2^n.
    """
    omega_start = -N//2
    omega_stop = N//2
    kappa_sigma = gauss_1d(-xi, sigma) / gauss_1d(0, sigma)
    normalisation = a**(-j)

    return [ normalisation * ( gauss_1d(a**j * omega - xi, sigma) - kappa_sigma*gauss_1d(a**j * omega, sigma) ) for omega in range(omega_start, omega_stop) ]


@numba.jit
def gauss_3d(x, y, z, sigma):
    return math.exp(-(x*x + y*y + z*z) / (2*sigma*sigma))


@numba.jit
def gauss_1d(x, sigma):
    return math.exp(-x*x / (2*sigma*sigma))


def plot(x):
    plt.plot(x)
    plt.show()


if __name__ == '__main__':
    dimensions = np.array([32, 32, 32])
    xi_radians = 4*np.pi/5
    xi = dimensions[0] * xi_radians/(2*np.pi)
    xi = np.array([xi, 0, 0])
    sigma_spatial = 0.6
    sigma_fourier = 1 / sigma_spatial
    n_points_fourier_sphere = 4
    rotation_matrices = rotation_matrices_fibonacci_spiral_unit_x(n_points_fourier_sphere)

    for j in range(3):
        for r in rotation_matrices:
            result = morlet_fourier_3d(dimensions, j, r, xi, sigma_fourier)
            print(result[dimensions[0]//2, dimensions[1]//2, dimensions[2]//2])
            plot3d(result)
