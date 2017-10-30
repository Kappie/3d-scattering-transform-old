import numpy as np
import tensorflow as tf
import sklearn as sk

from scattering_transform import scattering_transform


def load_data():
    """
    Returns CT scan data in numpy array of format n_samples x width x heigth x depth and
    label vector.
    """

    return 1

if __name__ == '__main__':
    # DATASET_PATH = r"D:\Geert\SOFT_TISSUE_THICK_DOWNSAMPLED_NUMPIFIED\dataset.npy"
    # LABELS_PATH = r"D:\Geert\SOFT_TISSUE_THICK_DOWNSAMPLED_NUMPIFIED\labels.npy"
    # AFFECTED = 1
    # UNAFFECTED = -1
    #
    # print("loading dataset with memory map")
    # dataset = np.load(DATASET_PATH, mmap_mode='r')
    # labels = np.load(LABELS_PATH)

    x = z = 64
    y = 128
    single_sample = np.random.rand(x, y, z)

    js = [0, 1]
    J = 2
    L = 2

    print("let's scatter. Shape of input is {}.".format(single_sample.shape))
    S = scattering_transform(single_sample, js, J, L)

    print(S.shape)
