import numpy as np
from IMLearn.base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics import loss_functions
from IMLearn.learners.classifiers.decision_stump import DecisionStump
import pandas as pd
import pickle

class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.models_ = []
        self.weights_ = np.zeros(self.iterations_)
        self.D_ = np.ones(len(y)) / len(y)
        for t in range(0, self.iterations_):
            new_model = self.wl_().fit(X, y * self.D_)
            self.models_.append(new_model)

            y_pred = new_model.predict(X)
            epsilon = np.sum(self.D_[y != y_pred])
            new_weight = 0.5 * np.log(epsilon**-1 - 1)
            self.weights_[t] = new_weight

            self.D_ = self.D_ * np.exp(-y * new_weight * y_pred)
            self.D_ = self.D_ / np.sum(self.D_)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        t = np.min([T, self.iterations_])
        return np.sign(np.sum(np.array([
                weight * model.predict(X)
                for weight, model in zip(self.weights_, self.models_[:t])
            ]), axis=0))

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        t = np.min([T, self.iterations_])
        return loss_functions.misclassification_error(y, self.partial_predict(X, T))

    def export_fitted(self, learners_path: str, weights_path: str, distribution_path: str):
        # pd.DataFrame(np.array(self.weights_)).to_csv(weights_path)
        # pd.DataFrame()
        pickle.dump(self.models_, open(learners_path, "wb"))
        pickle.dump(self.weights_, open(weights_path, "wb"))
        pickle.dump(self.D_, open(distribution_path, "wb"))

    def import_fitted(self, learners_path: str, weights_path: str, distribution_path: str):
        self.models_ = pickle.load(open(learners_path, "rb"))
        self.weights_ = pickle.load(open(weights_path, "rb"))
        self.D_ = pickle.load(open(distribution_path, "rb"))


if __name__ == "__main__":
    a = AdaBoost(DecisionStump, 5)
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    y = np.array([1, -1, 1])
    a.fit(X, y)
    a.predict(np.array([
        [1, 2, 3],
        [8, 10, 12]
    ]))
