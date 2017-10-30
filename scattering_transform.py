import numpy as np
import tensorflow as tf
from plot_slices import plot3d

from wavelets import filter_bank
from scattering import periodize


def scattering_transform(X, js, J, L):
    """
    X: input tensor in num_samples x width x height x depth format.
    js: length scales for filters. Filters will be dilated by 2**j for j in js.
    J: length scale used for averaging over scattered signals. (coefficients will be approximately translationally
    invariant over 2**J pixels.)
    L: number of angles for filters, spaced evenly in (0, pi).

    Computes only the first two layers of the scattering transform.
    """

    n_samples, width, height, depth = X.get_shape().as_list()
    filters = filter_bank(width, height, depth, js, J, L)
    psis = filters['psi']
    phis = filters['phi']
    scattering_coefficients = []
    transforms = []
    X_fourier = tf.fft3d(X)

    # First low-pass filter: Extract zeroth order coefficients
    zeroth_order_coefficents = tf.abs( tf.ifft3d(X_fourier * phis[0]) )
    # Downsample by factor 2**J
    zeroth_order_coefficents = downsample(zeroth_order_coefficents, 2**J)
    scattering_coefficients.append(zeroth_order_coefficents)

    for n1 in range(len(psis)):
        j1 = psis[n1]['j']

        # Calculate wavelet transform and apply modulus. Signal can be downsampled at 2**j1 without losing much energy.
        # See Bruna (2013).
        transform1 = tf.abs( tf.ifft3d(X_fourier * psis[n1][0]) )
        if j1 > 0:
            transform1 = downsample(transform1, 2**j1)
        # Cast back into complex64 is required for fft3d
        transform1 = tf.cast(transform1, tf.complex64)
        transform1_fourier = tf.fft3d(transform1)

        # Second low-pass filter: Extract first order coefficients.
        # The transform is already downsampled by 2**j1, so we take the version of phi that is downsampled by the same
        # factor. The scattering coefficients itself can be sampled at 2**J, so a downsampling of 2**(J - j1) remains.
        first_order_coefficients = tf.abs( tf.ifft3d(transform1_fourier * phis[j1]) )
        first_order_coefficients = downsample(first_order_coefficients, 2**(J-j1))
        scattering_coefficients.append(first_order_coefficients)

        for n2 in range(len(psis)):
            j2 = psis[n2]['j']
            if j1 < j2:
                # transform1 is already downsampled at 2**j1, so we take the wavelet that is downsampled at the same
                # factor.
                transform2 = tf.abs( tf.ifft3d(transform1_fourier * psis[n2][j1]) )
                # We can downsample transform2 at 2**j2, so here it remains to downsample with the factor 2**(j2-j1).
                transform2 = downsample(transform2, 2**(j2-j1))
                transform2 = tf.cast(transform2, tf.complex64)
                transform2_fourier = tf.fft3d(transform2)

                # Third low-pass filter. Extract second-order coefficients.
                # The transform is already downsampled by 2**j2, so we take the version of phi that is downsampled by
                # the same factor. The scattering coefficients itself can be sampled at 2**J, so a downsampling of
                # 2**(J - j2) remains.
                second_order_coefficients = tf.abs( tf.ifft3d(transform2_fourier * phis[j2]) )
                second_order_coefficients = downsample(second_order_coefficients, 2**(J-j2))
                scattering_coefficients.append(second_order_coefficients)

    scattering_coefficients = tf.concat(scattering_coefficients, axis=0)
    return scattering_coefficients


def downsample(signal, factor):
    """
    Signal NWHD format in real space.
    """
    signal_shape = signal.get_shape().as_list()
    # Append dummy dimension to indicate that number_of_channels = 1. (Required for pool operation.)
    signal = tf.reshape(signal, signal_shape + [1])
    window_size = [factor, factor, factor]
    strides =     [factor, factor, factor]
    if signal.dtype == tf.complex64 or signal.dtype == tf.complex128:
        real_part = tf.real(signal)
        imag_part = tf.imag(signal)
        downsampled_real_part = tf.nn.pool(real_part, window_size, "AVG", "VALID", strides=strides)
        downsampled_imag_part = tf.nn.pool(imag_part, window_size, "AVG", "VALID", strides=strides)
        downsampled_signal = tf.complex(downsampled_real_part, downsampled_imag_part)
    else:
        downsampled_signal = tf.nn.pool(signal, window_size, "AVG", "VALID", strides=strides)

    # Reshape back to original signal shape.
    new_signal_shape = [signal_shape[0], signal_shape[1]//factor, signal_shape[2]//factor, signal_shape[3]//factor]
    return tf.reshape(downsampled_signal, new_signal_shape)


js = [0, 1, 2]
J = 3
L = 2
X = tf.constant(np.array([np.random.random([8, 8, 8])]), dtype="complex64")

S = scattering_transform(X, js, J, L)

with tf.Session() as sess:
    result = sess.run(S)
    print(result)
    print(result.shape)
    # plot3d(np.absolute(result[0, :, :, :]))



# def __call__(self, x):
#     x_shape = x.get_shape().as_list()
#     x_h, x_w = x_shape[-2:]
#
#     if (x_w != self.N or x_h != self.M):
#         raise (RuntimeError('Tensor must be of spatial size (%i, %i)!' % (self.M, self.N)))
#
#     if (len(x_shape) != 4):
#         raise (RuntimeError('Input tensor must be 4D'))
#
#     J = self.J
#     phi = self.Phi
#     psi = self.Psi
#     n = 0
#
#     pad = self._pad
#     unpad = self._unpad
#
#     S = []
#     U_r = pad(x)
#
#     U_0_c = compute_fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
#     print(U_0_c, phi[0], U_r)
#     U_1_c = periodize(cdgmm(U_0_c, phi[0]), 2**J)
#     U_J_r = compute_fft(U_1_c, 'C2R')
#
#     S.append(unpad(U_J_r))
#     n = n + 1
#
#     for n1 in range(len(psi)):
#         j1 = psi[n1]['j']
#         U_1_c = cdgmm(U_0_c, psi[n1][0])
#         if j1 > 0:
#             U_1_c = periodize(U_1_c, k=2 ** j1)
#         U_1_c = compute_fft(U_1_c, 'C2C', inverse=True)
#         U_1_c = compute_fft(modulus(U_1_c), 'C2C')
#
#         # Second low pass filter
#         U_2_c = periodize(cdgmm(U_1_c, phi[j1]), k=2**(J - j1))
#         U_J_r = compute_fft(U_2_c, 'C2R')
#         S.append(unpad(U_J_r))
#         n = n + 1
#
#         for n2 in range(len(psi)):
#             j2 = psi[n2]['j']
#             if j1 < j2:
#                 U_2_c = periodize(cdgmm(U_1_c, psi[n2][j1]), k=2 ** (j2 - j1))
#                 U_2_c = compute_fft(U_2_c, 'C2C', inverse=True)
#                 U_2_c = compute_fft(modulus(U_2_c), 'C2C')
#
#                 # Third low pass filter
#                 U_2_c = periodize(cdgmm(U_2_c, phi[j2]), k=2 ** (J - j2))
#                 U_J_r = compute_fft(U_2_c, 'C2R')
#
#                 S.append(unpad(U_J_r))
#                 n = n + 1
#
#     if self.check:
#         return S
#
#     S = tf.concat(S, axis=1)
#     return S
