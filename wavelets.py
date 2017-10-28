import numpy as np
import tensorflow as tf
import scipy.signal
import scipy.fftpack as fft
from transforms3d.euler import euler2mat
from cube_show_slider import cube_show_slider
from itertools import product


def mother_gabor(u, xi, sigma):
    return np.exp(-np.dot(u, u)/(2*sigma**2) + 1j*np.dot(xi, u))


def gabor_filter(width, height, depth, j, alpha, beta, gamma, xi=np.array([3*np.pi/4, 0, 0]), sigma=1, a=2):
    """
    xi: modulation of periodic part of wavelet
    sigma: standard deviation of gaussian window
    a: scale parameter
    """
    rotation_matrix = euler2mat(alpha, beta, gamma, 'sxyz')

    gab_filter = np.zeros([width, height, depth], dtype=np.complex)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                centered_x, centered_y, centered_z = center(x, width), center(y, height), center(z, depth)
                rotated_u = np.dot(rotation_matrix, np.array([centered_x, centered_y, centered_z]))
                gab_filter[x, y, z] = (a**-j)*(1/sigma)*mother_gabor(rotated_u*a**-j, xi, sigma)

    return gab_filter


def gaussian_filter(width, height, depth, J):
    return np.real( gabor_filter(width, height, depth, J, 0, 0, 0, xi=np.array([0, 0, 0])) )


def gabor_filters(J, alphas, betas, gammas):
    return [ [ [ [ gabor_filter(j, alpha, beta, gamma) for gamma in gammas ] for beta in betas ] for alpha in alphas ] for j in range(J) ]


def center(index, list_length):
    return int(np.floor(index - (list_length-1)/2))


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
            for c in range(C):
                for i in range(int(2 ** res)):
                    for j in range(int(2 ** res)):
                        for k in range(int(2 ** res)):
                            crop[a, b, c] += x[a + i*A, b + j*B, c + k*C]

    return crop


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
            print("casting psi as constant tensor...")
            psi[resolution] = tf.constant(psi_signal_fourier_res, dtype='complex64')
            # What is the justification for this normalisation?
            psi[resolution] = tf.div(
                psi[resolution], (width * height * depth // 2**(2 * j)),
                name="psi_j%s_alpha%s_beta%s_gamma%s" % (j, alpha, beta, gamma))
            print("done.")

        filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = gaussian_filter(width, height, depth, J)
    phi_signal_fourier = fft.fftn(phi_signal)
    filters['phi']['j'] = J
    # We need the phi signal downsampled at all length scales j.
    for resolution in js:
        phi_signal_fourier_res = crop_freq_3d(phi_signal_fourier, resolution)
        print("casting phi as constant tensor...")
        filters['phi'][resolution] = tf.constant(phi_signal_fourier_res, dtype="complex64")
        filters['phi'][resolution] = tf.div(
            filters['phi'][resolution], (width * height * depth // 2**(2 * J)), name="phi_res%s" % resolution)
        print("done.")

    return filters
