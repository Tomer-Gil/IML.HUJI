from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product


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
        def get_most_frequent_label(labels: np.ndarray) -> int:
            values, labels_counts = np.unique(labels, return_counts=True)
            ind = np.argmax(labels_counts)
            return values[ind]

        def get_error(labels: np.ndarray, sign: int):
            return (labels != sign).sum()

        # X = np.c_[X, y]
        # lowest_error = np.inf
        # for i in range(X.shape[1]):
        #     for t in X[:, i].unique():
        #         # lowers, uppers = X[X[:, i] < t], X[X[:, i] >= t]
        #         # # _, lowers_labels_counts = np.unique(lowers[:, -1], return_counts=True)
        #         # # _, uppers_labels_counts = np.unique(uppers[:, -1], return_counts=True)
        #         # lowers_label, uppers_label = get_most_frequent_label(lowers[:, -1]), \
        #         #     get_most_frequent_label(uppers[:, -1])
        #         # err = get_error(lowers[:, -1], lowers_label) + get_error(uppers[:, -1], uppers_label)
        #
        #         if err < lowest_error:
        #             lowest_error = err
        #             self.j_ = i
        #             self.threshold_ = t
        #             self.sign_ =
        X = np.c_[X, y]
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                lowers, uppers = X[X[:, j] < t], X[X[:, j] >= t]
                for sign in [-1, 1]:
                    err = get_error(lowers[:, -1], sign) + get_error(uppers[:, -1], -sign)
                    if err < lowest_error:
                        lowest_error = err
                        self.j_ = j
                        self.threshold_ = t
                        self.sign_ = sign

        # err = np.inf
        # for j, sign in product(range(X.shape[1]), [-1, 1]):
        #     thr, thr_err = self._find_threshold(X[:, j], y, sign)
        #     if thr_err < err:
        #         self.threshold_, self.j_, self.sign_, err = thr, j, sign, thr_err
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
        raise NotImplementedError()

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
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
        ids = np.argsort(values)
        values, labels = values[ids], labels[ids]

        # Loss for classifying all as `sign` - namely, if threshold is smaller than values[0]
        loss = np.sum(np.abs(labels)[np.sign(labels) == sign])

        # Loss of classifying threshold being each of the values given
        loss = np.append(loss, loss - np.cumsum(labels * sign))

        id = np.argmin(loss)
        return np.concatenate([[-np.inf], values[1:], [np.inf]])[id], loss[id]

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
        raise NotImplementedError()

d = DecisionStump()
# d.fit(np.random.randn(20, 3), np.random.randint(-1, 2, 20))
# d.fit(np.random.randn(20, 3), np.array([1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1]))
d.fit(np.random.randn(5, 3), np.array([1, 1, -1, -1, 1]))
print()
