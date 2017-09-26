import numpy as np
import scipy.ndimage as ndim

def gaussian(u):
    xi = 0
    sigma = 1
    return np.exp(-np.dot(u, u)/2*sigma + 1j*np.dot(xi, u))

width, height, depth = 31, 31, 31
random_data = np.random.rand(width, height, depth)
result = ndim.filters.generic_filter(random_data, gaussian)

print(result)
