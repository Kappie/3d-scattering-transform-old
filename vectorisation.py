import numpy as np

from transforms3d.euler import euler2mat
from plot_slices import plot3d


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


def centered_array(size):
    """
    Returns [-size//2, ..., -2, -1, 0, 1, 2, ..., size//2]
    """
    return np.arange(size, dtype=np.float) - size // 2


alpha = 0
beta = 0
gamma = np.pi/2
j = 1
width = height = depth = 30

filter = gabor_filter(width, height, depth, j, alpha, beta, gamma)

plot3d(np.real(filter))
