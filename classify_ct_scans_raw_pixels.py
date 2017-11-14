import numpy as np
import pickle
import os
import datetime

from classify_ct_scans import classify
from my_utils import downsample
from visualisation_gradient import approximate_gradient
from plot_slices import plot3d


def load_downsampled_numpified_dataset(dataset_path, labels_path, n_samples_class="all", downsampling_res=2):
    dataset = np.load(DATASET_PATH, mmap_mode="c")
    labels = np.load(LABELS_PATH)
    if n_samples_class != "all":
        dataset = dataset[np.r_[:n_samples_class, -n_samples_class:0]]
        labels = labels[np.r_[:n_samples_class, -n_samples_class:0]]
    dataset = downsample_dataset(dataset, downsampling_res)
    return dataset, labels


def downsample_dataset(dataset, res):
    return np.ascontiguousarray(dataset[:, ::2**res, ::2**res, ::2**res])



def store_classifier(classifier):
    print("storing classifier: ", classifier)
    base_folder = "classifiers"
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "classifier_" + datetime_string + ".pickle"
    full_path = os.path.join(base_folder, file_name)
    with open(full_path, "wb") as handle:
        pickle.dump(classifier, handle)


if __name__ == '__main__':
    DATASET_PATH = r"F:\GEERT\DATASET_NUMPIFIED\dataset.npy"
    LABELS_PATH = r"F:\GEERT\DATASET_NUMPIFIED\labels.npy"
    AFFECTED = 1
    UNAFFECTED = -1
    n_samples_class = 10
    downsampling_res = 4

    dataset, labels = load_downsampled_numpified_dataset(DATASET_PATH, LABELS_PATH, n_samples_class=n_samples_class, downsampling_res=downsampling_res)
    n_samples, width, height, depth = dataset.shape
    dataset = dataset.reshape((n_samples, -1))
    classifier = classify(dataset, labels)
    # print(classifier.best_estimator_.decision_function(np.reshape(dataset[0], (1, -1))))

    gradient = approximate_gradient(classifier, dataset[0], dataset)
    gradient = np.reshape(gradient, (width, height, depth))
    plot3d(gradient)
