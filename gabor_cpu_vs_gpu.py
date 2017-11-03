import numpy as np
import math
import cmath
from numba import vectorize, cuda, jit
import time

from transforms3d.euler import euler2mat
from wavelets import (get_gabor_filter, get_gabor_filter_gpu, get_gaussian_filter,
    get_gaussian_filter_gpu, XI_DEFAULT, A_DEFAULT, SIGMA_DEFAULT)
from my_utils import get_blocks_and_threads


if __name__ == '__main__':
    x = z = 128
    y = 256
    j = 0
    alpha = 0
    beta = 0
    gamma = 0

    print("compile run")
    # get_gabor_filter_gpu(x, y, z, j, alpha, beta, gamma)
    # get_gabor_filter(x, y, z, j, alpha, beta, gamma)
    get_gaussian_filter(x, y, z, j)
    get_gaussian_filter_gpu(x, y, z, j)
    print("done")

    start = time.time()
    result_gpu = get_gaussian_filter_gpu(x, y, z, j)
    end = time.time()
    print("gaussian (gpu) took ", end - start)

    start = time.time()
    result_gpu = get_gaussian_filter(x, y, z, j)
    end = time.time()
    print("gaussian (cpu) took ", end - start)


    # start = time.time()
    # get_gaussian_filter(x, y, z, j)
    # end = time.time()
    # print("gaussian (compiled) took ", end - start)
