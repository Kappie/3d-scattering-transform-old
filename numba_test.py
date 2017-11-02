import math
import numpy as np
import pyculib.fft
import numba
from numba import cuda
from scipy.fftpack import fftn, ifftn
from my_utils import Multiply


def convolution_cpu(a, b):
    a_fourier = fftn(a)
    b_fourier = fftn(b)
    return ifftn(a_fourier*b_fourier)


def convolution_gpu(a, b):
    n_elements = np.prod(a.shape)
    a_gpu = cuda.to_device(a)
    b_gpu = cuda.to_device(b)
    product = cuda.device_array_like(a)
    convolution = cuda.device_array_like(a)
    a_fourier = cuda.device_array_like(a)
    b_fourier = cuda.device_array_like(b)

    pyculib.fft.fft(a, a_fourier)
    pyculib.fft.fft(b, b_fourier)
    product = Multiply(a_fourier, b_fourier)
    pyculib.fft.ifft(product, convolution)
    convolution = convolution.copy_to_host()
    return convolution / n_elements


def convolution_gpu_naive(a, b):
    n_elements = np.prod(a.shape)
    convolution = np.empty(a.shape).astype(np.complex64)
    a_fourier = np.empty(a.shape).astype(np.complex64)
    b_fourier = np.empty(a.shape).astype(np.complex64)
    pyculib.fft.fft(a, a_fourier)
    pyculib.fft.fft(b, b_fourier)
    pyculib.fft.ifft(a_fourier*b_fourier, convolution)
    return convolution / n_elements


def generate_random_arrays(x, y, z, seed=0):
    np.random.seed(seed)
    a_real = np.random.rand(x, y, z).astype(np.complex64)
    a_imag = np.random.rand(x, y, z).astype(np.complex64)
    b_real = np.random.rand(x, y, z).astype(np.complex64)
    b_imag = np.random.rand(x, y, z).astype(np.complex64)

    return a_real + 1j*a_imag, b_real + 1j*b_imag


def identity_cpu(a):
    return ifftn(fftn(a))


def identity_gpu(a):
    a_fourier = np.empty_like(a)
    result = np.empty_like(a)
    pyculib.fft.fft(a, a_fourier)
    pyculib.fft.ifft(a_fourier, result)
    return result / np.prod(a.shape)


def print_absolute_differences_real_imag(a, b):
    print("average absolute difference of real part")
    print(np.average(np.abs(np.real(a - b))))
    print("average absolute difference of imaginary part")
    print(np.average(np.abs(np.imag(a - b))))


def print_absolute_rel_differences_real_imag(a, b):
    print("avg abs rel difference of real part")
    print(np.average(np.abs(np.real(a - b)) / np.real(a)))
    print("avg abs rel difference of imag part")
    print(np.average(np.abs(np.imag(a - b)) / np.imag(a)))


x = y = 128
z = 256
a, b = generate_random_arrays(x, y, z)

# compilation run
gpu_result = Multiply(a, b)
