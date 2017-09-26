import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from wavelets import gabor_filter, gaussian_filter

# import pywt
# import pywt.data




def wavelet_transform(image, j, alpha, beta, gamma, dimensions=[5, 5, 5]):
    gab_filter = gabor_filter(j, alpha, beta, gamma, dimensions)
    result = convolve(image, gab_filter, mode="same")
    return result

def scattering_transform(image, J, alphas, betas, gammas, m):
    # js: scales, 0, ..., J
    # alphas, betas, gammas: positive angles
    # m: order of scattering transform

    transforms = {}
    coefficients = []
    for j in range(J):
        transforms[j] = []
        for a, alpha in enumerate(alphas):
            for b, beta in enumerate(betas):
                for c, gamma in enumerate(gammas):
                    transform = np.absolute(wavelet_transform(image, j, alpha, beta, gamma))
                    coefficient = gaussian_average(image, J)
                    transforms[j].append(transform)
                    coefficients.append(coefficient)

    if m > 1:
        transforms_higher_orders = []
        coefficients_higher_orders = []
        for transform in transforms:
            # TODO: not all js are needed in deeper layers
            result = scattering_transform(transform, J, alphas, betas, gammas, m-1)
            transforms_higher_orders.extend(result["transforms"])
            coefficients_higher_orders.extend(result["coefficients"])
        transforms.extend(transforms_higher_orders)
        coefficients.extend(coefficients_higher_orders)

    return {"transforms": transforms, "coefficients": coefficients}

def gaussian_average(image, J, dimensions=[5, 5, 5]):
    return np.average( convolve(image, gaussian_filter(J, dimensions)) )


# number of scales
J = 2
# number of directions per angle
L = 2
# order of transform
m = 2
# positive angles in uniform directions
alphas = betas = gammas = [(a/(L - 1))*np.pi for a in range(L)]

random_data = np.random.rand(21, 21, 21)

result = scattering_transform(random_data, J, alphas, betas, gammas, m)
print(len(result["coefficients"]))
print(result["coefficients"])
