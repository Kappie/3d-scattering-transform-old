import numpy as np
import scipy.signal
from transforms3d.euler import euler2mat
from cube_show_slider import cube_show_slider

def mother_gabor(u, xi, sigma):
    return np.exp(-np.dot(u, u)/2*sigma + 1j*np.dot(xi, u))

def gabor_filter(j, alpha, beta, gamma, dimensions, xi=np.array([3*np.pi/4, 0, 0]), sigma=1, a=2):
    """
    xi: modulation of periodic part of wavelet
    sigma: standard deviation of gaussian window
    a: scale parameter
    """
    # xi = np.array([0, 0, 0])
    rotation_matrix = euler2mat(alpha, beta, gamma, 'sxyz')

    width, height, depth = dimensions
    gab_filter = np.zeros([width, height, depth], dtype=np.complex)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                centered_x, centered_y, centered_z = center(x, width), center(y, height), center(z, depth)
                rotated_u = np.dot(rotation_matrix, np.array([centered_x, centered_y, centered_z]))
                gab_filter[x, y, z] = (a**-j)*(1/sigma)*mother_gabor(rotated_u*a**-j, xi, sigma)

    return gab_filter

def gabor_filters(js, alphas, betas, gammas, dimensions):
    return 0

def center(index, list_length):
    return int(np.floor(index - (list_length-1)/2))

filter = np.real(gabor_filter(3, np.pi/2, np.pi/2, 0, [41, 41, 11]))
# print(filter)
#
# cube_show_slider(filter)
