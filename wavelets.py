import numpy as np
import tensorflow as tf
import scipy.signal
from transforms3d.euler import euler2mat
from cube_show_slider import cube_show_slider
from itertools import product

def mother_gabor(u, xi, sigma):
    return np.exp(-np.dot(u, u)/(2*sigma**2) + 1j*np.dot(xi, u))

def gabor_filter(j, alpha, beta, gamma, max_dimension=int(40), xi=np.array([3*np.pi/4, 0, 0]), sigma=1, a=2):
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

def gabor_filters(J, alphas, betas, gammas):
    return [ [ [ [ gabor_filter(j, alpha, beta, gamma) for gamma in gammas ] for beta in betas ] for alpha in alphas ] for j in range(J) ]

def center(index, list_length):
    return int(np.floor(index - (list_length-1)/2))

def crop(data, lower_threshold=1e-4):
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


def crop_freq_3d(x, res):
    """
    Crop highest (1 - 2^-res) part of a fourier spectrum.
    (So for res = 1, cut highest half of the spectrum, res = 2 cut highest 3/4, etc.)
    Corresponds to a spatial downsampling of the image by a factor (res + 1).
    """
    M, N, O = x.shape[0], x.shape[1], x.shape[2]
    # Dimensions after cropping
    A, B, C = [int(dim // 2**res) for dim in [M, N, O]]

    crop = np.zeros((A, B, C), np.complex64)
    mask = np.ones(x.shape, np.float32)

    len_x, len_y, len_z = [int(dim * (1 - 2 ** (-res))) for dim in [M, N, O]]
    start_x, start_y, start_z = [int(dim * 2 ** (-res - 1)) for dim in [M, N, O]]
    # Crop highest frequencies
    mask[start_x:start_x + len_x, :, :] = 0
    mask[:, start_y:start_y + len_y, :] = 0
    mask[:, :, start_z:start_z + len_z] = 0
    x = np.multiply(x,mask)

    # Rescale spectrum? Or rather, rearrange the coefficients such that
    # the remaining ones fit into a smaller array, making use of the symmetry
    # of fourier spectrum of a real signal (e.g. an image).
    for a in range(A):
        for b in range(B):
            for c in range(C)
                for i in range(int(2 ** res)):
                    for j in range(int(2 ** res)):
                        for k in range(int(2 ** res)):
                            crop[a, b, c] += x[a + i*A, b + j*B, c + k*C]

    return crop
