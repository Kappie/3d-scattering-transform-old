import math
import numpy as np
import pyculib.fft
import numba
from numba import cuda
from scipy.fftpack import fftn, ifftn
import time


def time_me(f):
    def wrapper(*args):
        start = time.time()
        result = f(*args)
        end = time.time()
        print("{} took {} seconds".format(f.__name__, str(end - start)))
        return result

    return wrapper


@numba.vectorize(['complex64(complex64, complex64)'], target='cuda')
def Multiply(a, b):
    return a * b


@time_me
def convolution_cpu(a, b):
    a_fourier = fftn(a)
    b_fourier = fftn(b)
    return ifftn(a_fourier*b_fourier)


@time_me
def convolution_gpu(a, b):
    n_elements = np.prod(a.shape)
    a_gpu = cuda.to_device(a)
    b_gpu = cuda.to_device(b)
    product = cuda.device_array_like(a)
    convolution = cuda.device_array_like(a)
    a_fourier = cuda.device_array_like(a)
    b_fourier = cuda.device_array_like(b)

    pyculib.fft.fft(a_gpu, a_fourier)
    pyculib.fft.fft(b_gpu, b_fourier)
    product = Multiply(a_fourier, b_fourier)
    pyculib.fft.ifft(product, convolution)
    convolution = convolution.copy_to_host()
    return convolution / n_elements


@time_me
def convolution_gpu_inplace(a, b):
    n_elements = np.prod(a.shape)
    a_gpu = cuda.to_device(a)
    b_gpu = cuda.to_device(b)
    convolution = cuda.device_array_like(a)

    pyculib.fft.fft_inplace(a_gpu)
    pyculib.fft.fft_inplace(b_gpu)
    convolution = Multiply(a_gpu, b_gpu)
    pyculib.fft.ifft_inplace(convolution)
    convolution = convolution.copy_to_host()
    return convolution / n_elements


@time_me
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


if __name__ == '__main__':
    x = y = z = 256
    a, b = generate_random_arrays(x, y, z)

    print("Size of input: {}".format(a.shape))
    print("Compilation runs")
    convolution_gpu_naive(a, b)
    convolution_gpu(a, b)
    convolution_gpu_inplace(a, b)
    a, b = generate_random_arrays(x, y, z)
    convolution_cpu(a, b)
    print("done.")
    print("Real comparison single run:")
    convolution_gpu_naive(a, b)
    convolution_gpu(a, b)
    convolution_gpu_inplace(a, b)
    a, b = generate_random_arrays(x, y, z)
    convolution_cpu(a, b)
