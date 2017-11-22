import numpy as np
import numba
import math
import matplotlib.pyplot as plt
import scipy.fftpack
from itertools import product


def morlet_fourier(N, sigma_spatial, xi, n_periods):
    """
    Sigma is defined as the spatial standard deviation.
    N is assumed to be a power of two.
    n_periods is assumed to be odd.
    """
    # import pdb; pdb.set_trace()
    sigma_fourier = 1/sigma_spatial
    p_start = - (n_periods - 1)//2
    p_stop = (n_periods - 1)//2
    omega_start = -N//2 + p_start*N
    omega_stop = N//2 + p_stop*N - 1

    # Compute gaussians
    gauss_center = np.array([ gauss(omega-xi, sigma_fourier) for omega in range(omega_start, omega_stop+1) ])
    gauss_0 = np.array([ gauss(omega, sigma_fourier) for omega in range(omega_start + p_start*N, omega_stop + p_stop*N + 1) ])
    corrective_gaussians = np.array([ [ gauss_0[omega + p*N] for omega in range(N*n_periods)] for p in range(n_periods) ]).T

    # compute corrective factors kappa[p]
    b = np.array( [gauss(p*N - xi, sigma_fourier) for p in range(p_start, p_stop+1)] )
    A = np.array([[gauss((q-p)*N, sigma_fourier) for q in range(n_periods)] for p in range(n_periods)])
    corrective_factors = np.linalg.inv(A).dot(b)

    # reshape
    y = gauss_center - corrective_gaussians.dot(corrective_factors)
    y = np.reshape(y, (n_periods, N)).T
    y = np.squeeze(np.sum(y, axis=1))
    return y

    # y = gauss_center - corrective_gaussians * corrective_factors
    # y = reshape(y, N, nPeriods)
    # y = squeeze(sum(y, 2), 2)
    # print(y)


def morlet_fourier_naive(N, sigma_spatial, xi):
    sigma_fourier = 1/sigma_spatial
    omega_start = -N//2
    omega_stop = N//2

    kappa_sigma = gauss(-xi, sigma_fourier) / gauss(0, sigma_fourier)
    y = np.array([ gauss(omega - xi, sigma_fourier) - gauss(omega, sigma_fourier)*kappa_sigma  for omega in range(omega_start, omega_stop) ])
    return y



def plot(y):
    plt.plot(y)
    plt.show()

def plotxy(x, y):
    plt.plot(x, y)
    plt.show()


def gauss(x, sigma):
    return math.exp( -x*x/(2*sigma*sigma) )



if __name__ == '__main__':
    N = 2**8
    xi_radians = 2*np.pi/5
    xi = math.ceil( (xi_radians/(2*np.pi))*N )
    sigma_spatial = 0.8
    n_periods = 11

    morlet = morlet_fourier(N, sigma_spatial, xi, n_periods)
    morlet_naive = morlet_fourier_naive(N, sigma_spatial, xi)
    plot(morlet_naive)

    morlet_realspace = scipy.fftpack.ifft(morlet_naive)
    # plt.plot(morlet)
    # plt.show()
