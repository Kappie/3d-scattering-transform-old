import numpy as np
import tensorflow as tf
import scipy.fftpack as fft
import time
from plot_slices import plot3d

from wavelets import filter_bank
from my_utils import abs_after_convolve, extract_scattering_coefficients


def scattering_transform(X, js, J, L):
    """
    X: input image in width x height x depth format.
    js: length scales for filters. Filters will be dilated by 2**j for j in js.
    J: length scale used for averaging over scattered signals. (coefficients will be approximately translationally
    invariant over 2**J pixels.)
    L: number of angles for filters, spaced evenly in (0, pi).

    Computes only the first two layers of the scattering transform.
    """

    width, height, depth = X.shape
    X = X.astype(np.complex64)
    print("building filter bank...")
    start = time.time()
    filters = filter_bank(width, height, depth, js, J, L)
    end = time.time()
    print("done. Took {} seconds.".format(str(end - start)))
    start = time.time()
    psis = filters['psi']
    phis = filters['phi']
    scattering_coefficients = []
    transforms = []

    # First low-pass filter: Extract zeroth order coefficients
    zeroth_order_coefficients = extract_scattering_coefficients(X, phis[0], J)
    scattering_coefficients.append(zeroth_order_coefficients)

    for n1 in range(len(psis)):
        j1 = psis[n1]['j']

        # Calculate wavelet transform and apply modulus. Signal can be downsampled at 2**j1 without losing much energy.
        # See Bruna (2013).
        transform1 = abs_after_convolve(X, psis[n1][0], j1)

        # Second low-pass filter: Extract first order coefficients.
        # The transform is already downsampled by 2**j1, so we take the version of phi that is downsampled by the same
        # factor. The scattering coefficients itself can be sampled at 2**J, so a downsampling of 2**(J - j1) remains.
        first_order_coefficients = extract_scattering_coefficients(transform1, phis[j1], J - j1)
        scattering_coefficients.append(first_order_coefficients)

        for n2 in range(len(psis)):
            j2 = psis[n2]['j']
            if j1 < j2:
                # transform1 is already downsampled at 2**j1, so we take the wavelet that is downsampled at the same
                # factor.
                # We can downsample transform2 at 2**j2, so here it remains to downsample with the factor 2**(j2-j1).
                transform2 = abs_after_convolve(transform1, psis[n2][j1], j2 - j1)

                # Third low-pass filter. Extract second-order coefficients.
                # The transform is already downsampled by 2**j2, so we take the version of phi that is downsampled by
                # the same factor. The scattering coefficients itself can be sampled at 2**J, so a downsampling of
                # 2**(J - j2) remains.
                second_order_coefficients = extract_scattering_coefficients(transform2, phis[j2], J - j2)
                scattering_coefficients.append(second_order_coefficients)

    scattering_coefficients = np.array(scattering_coefficients)
    end = time.time()
    print("Done with scattering. Took {} seconds.".format(str(end - start)))
    return scattering_coefficients


if __name__ == '__main__':
    js = [0, 1, 2]
    J = 3
    L = 4
    x = y = 128
    z = 256
    X = np.random.rand(x, y, z)
    S = scattering_transform(X, js, J, L)
    print(S.shape)
