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
    DATASET_PATH = r"D:\Geert\SOFT_TISSUE_THICK_NUMPIFIED\dataset.npy"
    LABELS_PATH = r"D:\Geert\SOFT_TISSUE_THICK_NUMPIFIED\labels.npy"
    AFFECTED = 1
    UNAFFECTED = -1

    print("loading dataset with memory map")
    dataset = np.load(DATASET_PATH, mmap_mode='r')
    labels = np.load(LABELS_PATH)

    small_dataset = dataset[0:2]

    js = [0, 1]
    J = 2
    L = 2

    print("casting images as variable tensor... array of shape {}.".format(small_dataset.shape))
    X = tf.Variable(small_dataset, dtype=tf.complex64)

    print("let's scatter")
    S = scattering_transform(X, js, J, L)

    with tf.Session() as sess:
        result = sess.run(S)

    print(result.shape)
