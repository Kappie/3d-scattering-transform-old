import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds


class AffineSpaceApproximationClassifier:
    def __init__(self, d):
        """
        d: dimension of the approximation space.
        """
        self.d = d

    def fit(self, X):
        """
        X: {label: (N_label, D) array} dict, where N_label is the number of data points of a certain label and D is the number
        of features.
        """
        self.labels = X.keys
        self.X = X

        eigenvectors = {}
        singular_values = {}
        centroids = {}
        for label, data in X.items():
            projectors[label], singular_values[label], centroids[label] = self._projector_onto_subspace(data)

        self.eigenvectors = eigenvectors
        self.singular_values = singular_values
        self.centroids = centroids


    def predict(self, X):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples)
        for i in range(n_samples):
            feature_vector = X[i, :]
            label = self._predict_vector(feature_vector)

    def _predict_vector(self, x):
        approximation_errors = np.zeros(np.size(self.labels))
        for label in self.labels:
            




    def _projector_onto_subspace(self, X_label):
        """
        X_label: (n_samples, n_features) array
        """
        centroid = np.mean(X_label, axis=0)
        # Each row represents an observation
        covariance_matrix = np.cov(X_label, rowvar=False)
        U, s, U_transpose = svds(covariance_matrix, k=self.d)
        return U, s, centroid

def separate_dataset(X, labels, labels):
    """
    Given labeled dataset X of shape (n_samples, n_features), transform it into
    a dict with label names as keys and corresponding feature vectors as values.
    """
    result = {}
    for label in labels:
        indices = labels == label
        result[label] = X[indices]
    return result


if __name__ == '__main__':
    training_ratio = 0.8
    test_ratio = 1 - training_ratio
    digits = datasets.load_digits()

    feature_vectors = digits.data
    labels = digits.target
    n_training_set = int(training_ratio * len(feature_vectors))

    training_set = feature_vectors[:n_training_set, :]
    training_labels = labels[:n_training_set]
    labeled_training_set = separate_dataset(training_set, training_labels, digits.target_names)
    test_set = feature_vectors[n_training_set:, :]
    test_labels = labels[n_training_set:]

    d = 5
    classifier = AffineSpaceApproximationClassifier(d)
    decompositions = classifier.fit(labeled_training_set)
    print(decompositions[0].get_covariance().shape)
