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

    dataset = np.load(DATASET_PATH)
    labels = np.load(LABELS_PATH)

    small_dataset = dataset[0:2]
    print(small_dataset.shape)

    js = [0, 1]
    J = 2
    L = 2

    X = tf.constant(dataset, dtype="complex64")
    S = scattering_transform(X, js, J, L)

    with tf.Session() as sess:
        result = sess.run(S)

    print(result.shape)
