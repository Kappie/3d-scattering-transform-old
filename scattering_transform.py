import numpy as np
import tensorflow as tf
from plot_slices import plot3d

from wavelets import filter_bank


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

    zeroth_level_coefficents = perform_convolution(X_fourier, phis[0], J)
    scattering_coefficients.append(zeroth_level_coefficents)

    for n1 in range(len(psis)):
        j1 = psis[n1]['j']
        # Downsampling factor doesn't work for now.
        transforms = X_fourier * psis[n1][0]
        # supposed to be phis[j1] and downsampling factor j1
        first_order_coefficients = perform_convolution(transforms, phis[0], 0)
        scattering_coefficients.append(first_order_coefficients)

        for n2 in range(len(psis)):
            j2 = psis[n2]['j']
            if j1 < j2:
                1 + 1
                # print("{} < {} is a valid combination with alpha = {}, beta = {}, gamma = {}.".format(
                #     j1, j2, psis[n1]["alpha"], psis[n1]["beta"], psis[n1]["gamma"]))

    scattering_coefficients = tf.concat(scattering_coefficients, axis=0)
    return scattering_coefficients


def perform_convolution(signals, filter, downsampling_factor):
    """
    Remember that the signals and averaging filter are in Fourier space. 2 ** (downsampling
    factor) should be equal to the spatial window of the averaging filter.
    shape(signals) is num_samples x width x height x depth
    """
    # Computes 3d FFT over innermost 3 dimensions.
    convolution = tf.ifft3d(signals * filter)
    return convolution



js = [0, 1, 2]
J = 2
L = 5
X = tf.constant(np.array([np.random.random([10, 10, 10])]), dtype="complex64")

S = scattering_transform(X, js, J, L)

with tf.Session() as sess:
    result = sess.run(S)
    print(result)
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
