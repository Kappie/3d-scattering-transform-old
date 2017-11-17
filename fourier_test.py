import scipy.fftpack
import numpy as np
import math
import matplotlib.pyplot as plt


def cos_filter(N, xi):
    result = np.zeros(N)
    for n in range(N):
        n_centered = n - N // 2
        result[n] = math.cos(xi * n_centered * N / (2*math.pi))
    return result




if __name__ == '__main__':
    x = 128
    xi = 0
    cosine = cos_filter(x, xi)

    plt.plot(range(x), cosine)
    plt.show()
