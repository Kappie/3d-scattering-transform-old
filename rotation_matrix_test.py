import numpy as np
from transforms3d.euler import euler2mat
from itertools import product


L = 8
alphas = [0]
betas = gammas = [-np.pi/2 + np.pi * n / L for n in range(L)]

x = np.array([1, 0, 0])

for alpha, beta, gamma in product(alphas, betas, gammas):
    R = euler2mat(alpha, beta, gamma, 'sxyz')
    rotated_x = R.dot(x)
    print([alpha, beta, gamma], "gives ", np.round(rotated_x, decimals=1))


def print_rotated_x(x, R):
    rotated_x = R.dot(x)
    print(np.round(rotated_x, decimals=2))
