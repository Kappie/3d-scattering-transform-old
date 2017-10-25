import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds, eigs


class AffineSpaceApproximationClassifier:
    """
    See "Scattering Representations for Recognition" (PhD thesis J. Bruna), section 3.4 Generative Classification with Affine models.
    """
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
        self.labels = list(X.keys())
        self.X = X

        eigenvectors = {}
        eigenvalues = {}
        centroids = {}
        for label, data in X.items():
            eigenvectors[label], eigenvalues[label], centroids[label] = self._projector_onto_subspace(data)

        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.centroids = centroids


    def predict(self, X):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples)
        for i in range(n_samples):
            feature_vector = X[i, :]
            labels[i] = self._predict_vector(feature_vector)
        return labels


    def _predict_vector(self, x):
        approximation_errors = np.zeros(np.size(self.labels))
        for index, label in enumerate(self.labels):
            approximation_errors[index] = self._approximation_error(x, label)

        return self.labels[np.argmax(approximation_errors)]


    def _approximation_error(self, x, label):
        centroid = self.centroids[label]
        eigenvectors = self.eigenvectors[label]
        eigenvalues = self.eigenvalues[label]
        projector = np.linalg.multi_dot([eigenvectors, np.diag(eigenvalues), eigenvectors.T])
        x_centered = x - centroid
        approximation_error = np.linalg.norm(x_centered - np.dot(projector, x_centered))
        return approximation_error


    def _projector_onto_subspace(self, X_label):
        """
        X_label: (n_samples, n_features) array
        """
        centroid = np.mean(X_label, axis=0)
        # Each row represents an observation
        covariance_matrix = np.cov(X_label, rowvar=False)
        # U, eigenvalues, U_transpose = eigs(covariance_matrix, k=self.d)
        eigenvalues, eigenvectors = eigs(covariance_matrix, k=self.d)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        return eigenvectors, eigenvalues, centroid


def separate_dataset(X, target_labels, label_names):
    """
    Given labeled dataset X of shape (n_samples, n_features), transform it into
    a dict with label names as keys and corresponding feature vectors as values.
    """
    result = {}
    for label in label_names:
        indices = target_labels == label
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

    d = 2
    classifier = AffineSpaceApproximationClassifier(d)
    classifier.fit(labeled_training_set)
    predictions = classifier.predict(test_set)
    correct = predictions == test_labels
    percentage = np.count_nonzero(correct) / correct.shape[0]
    print(percentage)
    print(np.count_nonzero(correct))
    print(correct.shape)
