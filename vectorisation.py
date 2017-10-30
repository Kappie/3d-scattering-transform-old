import numpy as np

from transforms3d.euler import euler2mat
from plot_slices import plot3d


def gabor_vectorized(dimensions, j, alpha, beta, gamma, xi=np.array([3*np.pi/4, 0, 0]), a=2, sigma=1):
    """
    Outputs gabor filter of shape `dimensions`.
    """
    x = centered_array(dimensions[0])
    y = centered_array(dimensions[1])
    z = centered_array(dimensions[2])
    # Apply rotation matrix to coordinates and scale by a**j.
    R = euler2mat(alpha, beta, gamma, 'sxyz')
    x = (R[0, 0]*x + R[0, 1]*y + R[0, 2]*z) / a**j
    y = (R[1, 0]*x + R[1, 1]*y + R[1, 2]*z) / a**j
    z = (R[2, 0]*x + R[2, 1]*y + R[2, 2]*z) / a**j
    # Grid of coordinates for easy evaluation.
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    print(xx)
    # Apply gabor function and multiply by 1 / (a**j * sigma), as in the original
    # definition of scaled and rotated gabor.
    gabor_filter = (1/(a**j * sigma)) * np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2) + 1j*(x*xi[0] + y*xi[1] + z*xi[2]))
    return gabor_filter


def centered_array(size):
    """
    Returns [-size//2, ..., -2, -1, 0, 1, 2, ..., size//2]
    """
    return np.arange(size, dtype=np.float) - size // 2


alpha = np.pi/8
beta = np.pi/4
gamma = np.pi/2
j = 1
width = height = depth = 30

gabor_filter = gabor_vectorized([width, height, depth], j, alpha, beta, gamma)

plot3d(np.real(gabor_filter))
