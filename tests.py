import numpy as np
import scipy.fftpack

from scipy.misc import imread, imresize
from numba import cuda

from plot_slices import plot3d, plot2d
from wavelets import get_gabor_filter_gpu, get_gaussian_filter_gpu
from my_utils import (downsample, crop_freq_3d, crop_freq_3d_gpu, get_blocks_and_threads,
    abs_after_convolve, extract_scattering_coefficients)


def test_filters():
    x = y = z = 32
    j = 1
    alpha = 0
    beta = 0
    gamma = np.pi / 4

    gabor = get_gabor_filter_gpu(x, y, z, j, alpha, beta, gamma)
    gaussian = get_gaussian_filter_gpu(x, y, z, j)

    plot3d(np.real(gaussian))


def test_filters_fourier_support():
    x = y = z = 128
    j = 0
    alpha = 0
    beta = 0
    gamma = 0
    # xi = np.array([3*np.pi/4, 0, 0])
    xi = np.array([0, 0, 0])
    sigma = 5

    gabor = get_gabor_filter_gpu(x, y, z, j, alpha, beta, gamma, xi=xi, sigma=sigma)
    gabor_fourier = scipy.fftpack.fftn(gabor)
    gaussian = get_gaussian_filter_gpu(x, y, z, j)
    gaussian_fourier = scipy.fftpack.fftn(gaussian)
    plot3d(np.abs(gabor_fourier))
    plot3d(np.real(gabor))
    # plot3d(np.abs(gaussian_fourier))
    # plot3d(np.real(gaussian))


def get_test_3d_image():
    # Make a copy of 4 resized mona lisas to serve as test 3d image.
    image = imread("mona_lisa.gif", mode="F")
    image = imresize(image, (32, 32))
    image = np.stack([image, image, image, image], axis=2)
    return image


def test_downsampling_real_space():
    image = get_test_3d_image()
    res = 1
    image = downsample(image, res)
    plot3d(image)


def test_downsampling_fourier():
    image = get_test_3d_image()
    res = 1
    image_fourier = scipy.fftpack.fftn(image)
    image_fourier_downsampled = crop_freq_3d(image_fourier, res)
    image_downsampled = np.absolute( scipy.fftpack.ifftn(image_fourier_downsampled) )
    plot3d(image_downsampled)


def test_downsampling_fourier_gpu():
    image = get_test_3d_image()
    x, y, z = image.shape
    res = 1
    image_fourier = scipy.fftpack.fftn(image)

    image_fourier_gpu = cuda.to_device(image_fourier)
    result = cuda.device_array((x//2**res, y//2**res, z//2**res), dtype=np.complex64)
    blockspergrid, threadsperblock = get_blocks_and_threads(result.shape[0], result.shape[1], result.shape[2])
    image_fourier_downsampled = crop_freq_3d_gpu[blockspergrid, threadsperblock](image_fourier_gpu, result)
    result = result.copy_to_host()

    image_downsampled = np.absolute( scipy.fftpack.ifftn(result) )
    plot3d(image_downsampled)


def test_abs_after_convolve():
    x = y = z = 32
    j = 0
    delta = np.zeros((x, y, z)).astype(np.complex64)
    delta[0, 0, 0] = 1
    A = np.random.rand(x, y, z).astype(np.complex64)
    abs_convolution = abs_after_convolve(delta, A, j)
    # Convolution with a delta should be equal to the input signal.
    print(np.average(np.abs(np.abs(A) - abs_convolution)))


if __name__ == '__main__':
    # test_filters()
    # test_downsampling_real_space()
    # test_downsampling_fourier()
    # test_downsampling_fourier_gpu()
    # test_abs_after_convolve()
    test_filters_fourier_support()
