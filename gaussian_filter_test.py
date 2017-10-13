from wavelets import gaussian_filter, gabor_filter
from plot_slices import plot_slices, plot_2d_array
from filters_bank import crop_freq, morlet_2d
import numpy as np
import scipy.fftpack as fft
import scipy.ndimage as ndim
from scipy.misc import imshow
import matplotlib.pyplot as plt



def middle_hole_mask(x):
    hole_size = 126
    M, N = x.shape
    mask = np.ones([M, N])
    mask[int((M - hole_size)/2):int((M + hole_size)/2), int((N - hole_size)/2):int((N + hole_size)/2)] = 0
    return mask



def crop_high_frequencies(x):
    mask = middle_hole_mask(x)
    return np.multiply(x, mask)

# 128 x 128
image = ndim.imread("mona_lisa.gif", flatten=True)
fourier_image = fft.fft2(image)
cropped_fourier_image = crop_high_frequencies(fourier_image)
cropped_fourier_image = crop_freq(fourier_image, 3)
reconstruction = np.absolute( fft.ifft2(cropped_fourier_image) )
plt.figure()
plt.imshow(np.log(np.absolute(fft.fft2(reconstruction))))
plt.show()

# inverse_fourier_image = fft.ifft2(fourier_image)
# plot_2d_array(np.log(np.absolute(fourier_image)))
# plot_2d_array(np.real(inverse_fourier_image))

# print(inverse_fourier_image)
