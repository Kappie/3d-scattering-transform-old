import numpy as np
import scipy.fftpack

from plot_slices import plot3d, plot2d


def plot_fourier_spectrum(image):
    fourier_image = scipy.fftpack.fftn(image)
    plot3d(np.abs(fourier_image))


if __name__ == '__main__':
    DATASET_PATH = r"F:\GEERT\DATASET_NUMPIFIED\dataset.npy"
    dataset = np.load(DATASET_PATH, mmap_mode="r")

    plot_fourier_spectrum(dataset[1, :, :, :])
    plot_fourier_spectrum(dataset[2, :, :, :])
    plot_fourier_spectrum(dataset[3, :, :, :])
    plot_fourier_spectrum(dataset[4, :, :, :])
