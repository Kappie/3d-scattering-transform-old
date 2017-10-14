import matplotlib
from wavelets import gaussian_filter, gabor_filter, crop_freq_3d
from plot_slices import plot3d, plot2d
from filters_bank import crop_freq, morlet_2d
import numpy as np
import scipy.fftpack as fft
import scipy.ndimage as ndim
import scipy.misc
import matplotlib.pyplot as plt


# 128 x 128
image = ndim.imread("mona_lisa.gif", flatten=True)
fourier_image = fft.fft2(image)
inverse_fourier_image = fft.ifft2(fourier_image)

plot2d(np.absolute(inverse_fourier_image))
# image = np.swapaxes(np.array([image, image, image, image]), 0, 2)
# image = np.swapaxes(image, 0, 1)

# res = 1
# fourier_image = fft.fftn(image)
# cropped_fourier_image = crop_freq_3d(fourier_image, res)
# reconstruction = np.absolute( fft.ifftn(cropped_fourier_image) )
#
# plot_2d_array(reconstruction[:, :, 0])


# plt.figure()
# plt.imshow(reconstruction[:, :, 0])
# plt.show()



# fourier_image = fft.fft2(image)
# cropped_fourier_image = crop_high_frequencies(fourier_image)
# cropped_fourier_image = crop_freq(fourier_image, 3)
# reconstruction = np.absolute( fft.ifft2(cropped_fourier_image) )
# plt.figure()
# plt.imshow(np.log(np.absolute(fft.fft2(reconstruction))))
# plt.show()

# inverse_fourier_image = fft.ifft2(fourier_image)
# plot_2d_array(np.log(np.absolute(fourier_image)))
# plot_2d_array(np.real(inverse_fourier_image))

# print(inverse_fourier_image)
