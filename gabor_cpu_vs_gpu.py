import numpy as np
import math
import cmath
from numba import vectorize, cuda, jit
import time

from transforms3d.euler import euler2mat
from wavelets import gabor_filter
from my_utils import get_blocks_and_threads


XI_DEFAULT = np.array([3*np.pi/4, 0, 0])
A_DEFAULT = 2
SIGMA_DEFAULT = 1.


def get_gabor_filter_gpu(width, height, depth, j, alpha, beta, gamma, xi=XI_DEFAULT, a=A_DEFAULT, sigma=SIGMA_DEFAULT):
    R = euler2mat(alpha, beta, gamma, 'sxyz')
    result = np.empty((width, height, depth), dtype=np.complex64)
    blockspergrid, threadsperblock = get_blocks_and_threads(width, height, depth)
    gabor_filter_gpu[blockspergrid, threadsperblock](width, height, depth, j, R, xi, a, sigma, result)
    return result


def get_gabor_filter(width, height, depth, j, alpha, beta, gamma, xi=XI_DEFAULT, a=A_DEFAULT, sigma=SIGMA_DEFAULT):
    R = euler2mat(alpha, beta, gamma, 'sxyz')
    result = np.empty((width, height, depth), dtype=np.complex64)
    gabor_filter_compiled(width, height, depth, j, R, xi, a, sigma, result)
    return result


def get_gaussian_filter(width, height, depth, J, a=A_DEFAULT, sigma=SIGMA_DEFAULT):
    result = np.empty((width, height, depth), dtype=np.float32)
    gaussian_filter_compiled(width, height, depth, J, a, sigma, result)
    return result


@jit(['void(int64, int64, int64, int64, float64[:, :], float64[:], int64, float64, complex64[:, :, :])'], nopython=True)
def gabor_filter_compiled(width, height, depth, j, R, xi, a, sigma, result):
    scale_factor = 1/(sigma*a**j)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                x_prime = (R[0, 0]*x + R[0, 1]*y + R[0, 2]*z) * scale_factor
                y_prime = (R[1, 0]*x + R[1, 1]*y + R[1, 2]*z) * scale_factor
                z_prime = (R[2, 0]*x + R[2, 1]*y + R[2, 2]*z) * scale_factor
                result[x, y, z] = scale_factor * cmath.exp(-(x_prime**2 + y_prime**2 + z_prime**2)/2 + 1j*sigma*(x_prime*xi[0] + y_prime*xi[1] + z_prime*xi[2]))

@cuda.jit()
def gabor_filter_gpu(width, height, depth, j, R, xi, a, sigma, result):
    x, y, z = cuda.grid(3)
    scale_factor = 1/(sigma*a**j)
    if x < width and y < height and z < depth:
        x_prime = (R[0, 0]*x + R[0, 1]*y + R[0, 2]*z) * scale_factor
        y_prime = (R[1, 0]*x + R[1, 1]*y + R[1, 2]*z) * scale_factor
        z_prime = (R[2, 0]*x + R[2, 1]*y + R[2, 2]*z) * scale_factor
        result[x, y, z] = scale_factor * cmath.exp(-(x_prime**2 + y_prime**2 + z_prime**2)/2 + 1j*sigma*(x_prime*xi[0] + y_prime*xi[1] + z_prime*xi[2]))


@jit(['void(int64, int64, int64, int64, int64, float64, float32[:, :, :])'], nopython=True)
def gaussian_filter_compiled(width, height, depth, j, a, sigma, result):
    scale_factor = 1/(sigma*a**j)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                result[x, y, z] = scale_factor * math.exp(-(x**2 + y**2 + z**2)*scale_factor/2)


if __name__ == '__main__':
    x = z = 128
    y = 256
    j = 0
    alpha = 0
    beta = 0
    gamma = 0

    print("compile run")
    get_gabor_filter_gpu(x, y, z, j, alpha, beta, gamma)
    get_gabor_filter(x, y, z, j, alpha, beta, gamma)
    # get_gaussian_filter(x, y, z, j)
    # get_gaussian_filter(x, y, z, j)
    print("done")

    start = time.time()
    result_gpu = get_gabor_filter_gpu(x, y, z, j, alpha, beta, gamma, xi=np.array([0., 0., 0.]))
    end = time.time()
    print("gabor (gpu) took ", end - start)

    start = time.time()
    result_cpu = get_gabor_filter(x, y, z, j, alpha, beta, gamma, xi=np.array([0., 0., 0.]))
    end = time.time()
    print("gabor (cpu) took ", end - start)

    print(result_gpu - result_cpu)


    # start = time.time()
    # get_gaussian_filter(x, y, z, j)
    # end = time.time()
    # print("gaussian (compiled) took ", end - start)
