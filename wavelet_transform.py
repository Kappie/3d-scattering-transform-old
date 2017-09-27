import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.signal import convolve
from wavelets import gabor_filter, gaussian_filter
from collections import namedtuple

# import pywt
# import pywt.data

Transform = namedtuple("WaveletTransform", ["transform", "js", "alphas", "betas", "gammas", "scattering_coefficient"])

def wavelet_transform(image, j, alpha, beta, gamma):
    gab_filter = gabor_filter(j, alpha, beta, gamma)
    result = convolve(image, gab_filter, mode="same")
    return result

def scattering_transform(image, J, alphas, betas, gammas, m):
    transforms = [[Transform(image, [], [], [], [], gaussian_average(image, J))]]

    for order in range(m):
        transforms_previous_layer = transforms[-1]
        transforms_current_layer = []
        for transform in transforms_previous_layer:
            # For the second and higher layers, only scales j < j_max are relevant,
            # where j_max is the length scale of the previous wavelet transform. (See Bruna 2013.)
            if transform.js == []:
                j_max = J
            else:
                j_max = transform.js[-1]

            for j, alpha, beta, gamma in product(range(j_max), alphas, betas, gammas):
                next_transform = np.absolute(wavelet_transform(transform.transform, j, alpha, beta, gamma))
                scattering_coefficient = gaussian_average(next_transform, J)
                transforms_current_layer.append(Transform(
                    next_transform,
                    transform.js + [j],
                    transform.alphas + [alpha],
                    transform.betas + [beta],
                    transform.gammas + [gamma],
                    scattering_coefficient))
                print("appended j:{0}, alpha: {1}, beta: {2}, gamma: {3} at order {4}".format(j, alpha, beta, gamma, order))

        transforms.append(transforms_current_layer)

    return transforms

def gaussian_average(image, J):
    return np.average( convolve(image, gaussian_filter(J)) )

# number of scales
J = 2
# number of directions per angle
L = 2
# order of transform
m = 2
# positive angles in uniform directions
alphas = betas = gammas = [(a/(L - 1))*np.pi for a in range(L)]

random_data = np.random.rand(31, 31, 31)

result = scattering_transform(random_data, J, alphas, betas, gammas, m)
print(len(result[1]))
print(len(result[2]))
