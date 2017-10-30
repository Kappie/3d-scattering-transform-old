import numpy as np
import tensorflow as tf
import scipy.fftpack as fft
import time
from plot_slices import plot3d

from wavelets import filter_bank
from my_utils import fourier, inverse_fourier, crop_freq_3d


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
    print("building filter bank...")
    start = time.time()
    filters = filter_bank(width, height, depth, js, J, L)
    end = time.time()
    print("done. Took {} seconds.".format(str(end - start)))
    psis = filters['psi']
    phis = filters['phi']
    scattering_coefficients = []
    transforms = []
    X_fourier = fourier(X)

    # First low-pass filter: Extract zeroth order coefficients
    zeroth_order_coefficients_fourier = X_fourier * phis[0]
    # Downsample by factor 2**J
    zeroth_order_coefficients_fourier = crop_freq_3d(zeroth_order_coefficients_fourier, J)
    # Transform back to real space.
    zeroth_order_coefficients = np.abs( inverse_fourier(zeroth_order_coefficients_fourier) )
    scattering_coefficients.append(zeroth_order_coefficients)

    for n1 in range(len(psis)):
        j1 = psis[n1]['j']

        # Calculate wavelet transform and apply modulus. Signal can be downsampled at 2**j1 without losing much energy.
        # See Bruna (2013).
        # transform1 = np.abs( fft.ifftn(X_fourier * psis[n1][0]) )
        # if j1 > 0:
        #     transform1 = downsample(transform1, 2**j1)
        # # Cast back into complex64 is required for fft3d
        # # transform1 = tf.cast(transform1, tf.complex64)
        # transform1_fourier = fft.fftn(transform1)

        transform1 = np.abs(inverse_fourier( crop_freq_3d( X_fourier * psis[n1][0], j1 ) ))
        transform1_fourier = fourier(transform1)

        # Second low-pass filter: Extract first order coefficients.
        # The transform is already downsampled by 2**j1, so we take the version of phi that is downsampled by the same
        # factor. The scattering coefficients itself can be sampled at 2**J, so a downsampling of 2**(J - j1) remains.
        # first_order_coefficients = np.abs( fft.ifftn(transform1_fourier * phis[j1]) )
        # first_order_coefficients = downsample(first_order_coefficients, 2**(J-j1))
        first_order_coefficients = np.abs(inverse_fourier( crop_freq_3d(transform1_fourier * phis[j1], J - j1) ))
        scattering_coefficients.append(first_order_coefficients)

        for n2 in range(len(psis)):
            j2 = psis[n2]['j']
            if j1 < j2:
                # # transform1 is already downsampled at 2**j1, so we take the wavelet that is downsampled at the same
                # # factor.
                # transform2 = np.abs( fft.ifftn(transform1_fourier * psis[n2][j1]) )
                # # We can downsample transform2 at 2**j2, so here it remains to downsample with the factor 2**(j2-j1).
                # transform2 = downsample(transform2, 2**(j2-j1))
                # # transform2 = tf.cast(transform2, tf.complex64)
                # transform2_fourier = fft.fftn(transform2)

                transform2 = np.abs(inverse_fourier( crop_freq_3d( transform1_fourier * psis[n2][j1], j2 - j1 ) ))
                transform2_fourier = fourier(transform2)

                # Third low-pass filter. Extract second-order coefficients.
                # The transform is already downsampled by 2**j2, so we take the version of phi that is downsampled by
                # the same factor. The scattering coefficients itself can be sampled at 2**J, so a downsampling of
                # 2**(J - j2) remains.
                # second_order_coefficients = np.abs( fft.ifftn(transform2_fourier * phis[j2]) )
                # second_order_coefficients = downsample(second_order_coefficients, 2**(J-j2))

                second_order_coefficients = np.abs(inverse_fourier( crop_freq_3d(transform2_fourier * phis[j2], J - j2) ))
                scattering_coefficients.append(second_order_coefficients)

    scattering_coefficients = np.array(scattering_coefficients)
    return scattering_coefficients




# def downsample(data, factor):
#     """
#     Simple spatial downsampling.
#     """
#     return data[::factor, ::factor, ::factor]


if __name__ == '__main__':
    js = [0, 1, 2]
    J = 2
    L = 2
    x = y = z = 16
    X = np.random.rand(x, y, z)
    S = scattering_transform(X, js, J, L)
    print(S)
    print(S.shape)
