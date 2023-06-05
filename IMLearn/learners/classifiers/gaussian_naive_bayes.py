from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y).astype(np.int)
        self.pi_ = np.array([
            len(X[y == k]) / len(X) for k in self.classes_
        ])
        self.mu_ = np.array([
            [np.mean(X[y == k, j]) for j in range(X.shape[1])] for k in self.classes_
        ])
        self.vars_ = np.array([
            np.diag([np.mean((X[y == k, j] - self.mu_[k][j]) ** 2) for j in range(X.shape[1])]) for k in self.classes_
        ])


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.array([
            np.argmax(x)
            for x in self.likelihood(X)
        ])

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        def normal_dist_prob(x, mu, cov_mat):
            return ((2 * np.pi) ** len(x) * np.linalg.det(cov_mat)) ** 0.5 * np.exp(
                -1/2 * np.einsum("i,j,ij->", (x - mu), (x - mu), np.linalg.inv(cov_mat))
            )
        # The einstein sum is equivalent to
        # np.einsum("i, ij, j->", (X[0] - self.mu_[0]), self.vars_[0], (X[0] - self.mu_[0]))
        # np.einsum("i, ij, j->", (x - mu), np.linalg.inc(cov_mat), (x - mu))
        return np.array([
            [
                self.pi_[j]
                * normal_dist_prob(x, self.mu_[j], self.vars_[j])
                for j in self.classes_
            ] for x in X
        ])

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
