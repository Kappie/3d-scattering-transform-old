import numpy as np
import pyculib.fft


@numba.vectorize(['complex64(complex64, complex64)'], target='cuda')
def Multiply(a, b):
    return a * b


@numba.vectorize(['complex64(complex64, complex64)'], target='cuda')
def Modulus(x):
    return abs(x)


def fourier(signal):
    signal = signal.astype(np.complex64)
    signal_fourier = np.empty_like(signal)
    pyculib.fft.fft(signal, signal_fourier)
    return signal_fourier


def inverse_fourier(signal_fourier):
    signal = np.empty_like(signal_fourier)
    pyculib.fft.ifft(signal_fourier, signal)
    return signal


def crop_freq_3d(x, res):
    """
    Crop highest (1 - 2^-res) part of a fourier spectrum.
    (So for res = 1, cut highest half of the spectrum, res = 2 cut highest 3/4, etc.)
    This comes down to only taking the dim/(2**(res+1)) elements at the front and end of each dimension of the original array.
    In 2D, for res = 1 and a 4x4 input, you would get (taking only the single element at the front and back of each dimension)
    [[a00 a03], [a30, a33]]
    Corresponds to a spatial downsampling of the image by a factor (res + 1).
    Expects dimensions of array to be powers of 2.
    """
    if res == 0:
        return x

    M, N, O = x.shape[0], x.shape[1], x.shape[2]
    end_x, end_y, end_z = [int(dim * 2 ** (-res - 1)) for dim in [M, N, O]]
    indices_x, indices_y, indices_z = [ list(range(end_index)) + list(range(-end_index, 0)) for end_index in [end_x, end_y, end_z] ]
    indices = np.ix_(indices_x, indices_y, indices_z)
    return x[indices]


def modulus_after_inverse_fourier(signal):
    n_elements = np.prod(signal.shape)
    signal_gpu = cuda.to_device(signal)
    signal_inverse_fourier = cuda.device_array_like(signal)

    pyculib.fft.fft(signal_gpu, signal_inverse_fourier)
    modulus = Modulus(signal_inverse_fourier)

    return modulus / n_elements
