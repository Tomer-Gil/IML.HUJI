from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics import loss_functions


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        lowest_error = np.inf  # Equivalent to lowest_error = 1
        for j, sign in product(range(X.shape[1]), [-1, 1]):
            threshold, threshold_error = self._find_threshold(X[:, j], y, sign)
            if threshold_error < lowest_error:
                lowest_error = threshold_error
                self.j_ = j
                self.threshold_ = threshold
                self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.array([
            -self.sign if sample[self.j_] < self.threshold_ else self.sign_ for sample in X
        ])

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        def get_error(labels: np.ndarray, sign: int):
            return (labels != sign).sum()
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        temp_error = np.inf
        for t in np.concatenate([[-np.inf], np.unique(values), [np.inf]]):
            lowers, uppers = labels[values < t], labels[values >= t]
            err = get_error(lowers, -sign) + get_error(uppers, sign)
            if err < temp_error:
                temp_threshold = t
                temp_error = err
        return temp_threshold, temp_error / len(values)

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
        return loss_functions.misclassification_error(y, self.predict(X))


def test_find_threshold():
    d = DecisionStump()
    d._find_threshold(np.array([1.5, 7, 4.33]), np.array([1, -1, 1]), -1)


if __name__ == "__main__":
    d = DecisionStump()
    # d.fit(np.random.randn(20, 3), np.random.randint(-1, 2, 20))
    # d.fit(np.random.randn(20, 3), np.array([1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1]))
    d.fit(np.random.randn(5, 3), np.array([1, 1, -1, -1, 1]))
    print()

    test_find_threshold()