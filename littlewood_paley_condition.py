import numpy as np
import scipy.fftpack
import numba

from wavelets import filter_bank
from my_utils import downsample

def littlewood_paley_condition(phi, psis):
    abs_squared_psis = [ abs_squared(psi) for psi in psis ]
    abs_squared_phi = abs_squared(phi)
    # abs_squared_psis_indices_flipped = [np.flip(np.flip(np.flip(abs_squared_psi, axis=0), axis=1), axis=2) for abs_squared_psi in abs_squared_psis]
    print(abs_squared_phi[0, 0, 0])
    psi_sum = np.zeros(phi.shape)
    for n in range(len(abs_squared_psis)):
        psi_sum += abs_squared_psis[n] #+ abs_squared_psis_indices_flipped[n]

    return abs_squared_phi + 0.5*psi_sum


def littlewood_paley_sum(phi, psis, omega):
    """
    Get Littlewood-Paley sum for frequency omega. Supposes phi and psis are already in fourier space.
    """
    k, l, m = omega[0], omega[1], omega[2]
    return abs(phi[k, l, m])**2 + 0.5 * sum( [abs(psi[k, l, m])**2 for psi in psis] )


@numba.vectorize([numba.float32(numba.complex64),numba.float64(numba.complex128)])
def abs_squared(x):
    return x.real**2 + x.imag**2


if __name__ == '__main__':

    y = 32
    x = z = 16
    js = [0, 1, 2]
    J = js[-1]
    n_points_fourier_sphere = 10
    sigma = 1
    xi = np.array([4*np.pi/5, 0., 0.])

    filters = filter_bank(x, y, z, js, J, n_points_fourier_sphere, sigma, xi)

    # Get original, undownsampled filters (resolution 0) in fourier space. Phi is already in Fourier space.
    phi = filters['phi'][0]
    psis = [scipy.fftpack.fftn(psi[0]) for psi in filters['psi']]
    lp_condition = downsample(littlewood_paley_condition(phi, psis), J)
    print(lp_condition)
    print(sigma)
