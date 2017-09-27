import numpy as np
import scipy.signal
from transforms3d.euler import euler2mat
from cube_show_slider import cube_show_slider

def mother_gabor(u, xi, sigma):
    return np.exp(-np.dot(u, u)/(2*sigma**2) + 1j*np.dot(xi, u))

def gabor_filter(j, alpha, beta, gamma, max_dimension=int(11), xi=np.array([3*np.pi/4, 0, 0]), sigma=1, a=2):
    """
    xi: modulation of periodic part of wavelet
    sigma: standard deviation of gaussian window
    a: scale parameter
    """
    # xi = np.array([0, 0, 0])
    rotation_matrix = euler2mat(alpha, beta, gamma, 'sxyz')

    width = height = depth = max_dimension
    gab_filter = np.zeros([width, height, depth], dtype=np.complex)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                centered_x, centered_y, centered_z = center(x, width), center(y, height), center(z, depth)
                rotated_u = np.dot(rotation_matrix, np.array([centered_x, centered_y, centered_z]))
                gab_filter[x, y, z] = (a**-j)*(1/sigma)*mother_gabor(rotated_u*a**-j, xi, sigma)

    return crop(gab_filter)

def gaussian_filter(J):
    return np.real( gabor_filter(J, 0, 0, 0, xi=np.array([0, 0, 0])) )

def gabor_filters(js, alphas, betas, gammas, dimensions):
    return 0

def center(index, list_length):
    return int(np.floor(index - (list_length-1)/2))

def crop(data, lower_threshold=1e-5):
    """
    Crops array of complex numbers if absolute value is smaller than lower_threshold
    """
    for i in range(data.ndim):
        data = np.swapaxes(data, 0, i)  # send i-th axis to front
        while np.all( np.absolute(data)[0]<lower_threshold ):
            data = data[1:]
        while np.all( np.absolute(data)[-1]<lower_threshold ):
            data = data[:-1]
        data = np.swapaxes(data, 0, i)  # send i-th axis to its original position
    return data
