from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from IMLearn.metrics import loss_functions
from numpy.linalg import svd


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        X = np.c_[np.ones(len(X)), X] if self.include_intercept_ else X
        U, Sigma_Dagger, V_T = svd(X, full_matrices=False)
        Sigma_lambda = np.diag(Sigma_Dagger / (Sigma_Dagger ** 2 + self.lam_))
        self.coefs_ = V_T.T @ Sigma_lambda @ U.T @ y

        # X = np.c_[np.ones(len(X)), X] if self.include_intercept_ else X
        # X = np.r_[X, np.diag(np.full(X.shape[1], self.lam_))]
        # y = np.r_[y, np.zeros(len(X) - len(y))]
        # self.coefs_ = np.linalg.pinv(X) @ y


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
        X = np.c_[np.ones(len(X)), X] if self.include_intercept_ else X
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return loss_functions.mean_square_error(y, self._predict(X))


if __name__ == "__main__":
    model = RidgeRegression(5)
    model.fit(np.array([
        [1, 2, 3],
        [4, 5, 6]
    ]), np.array([8, 9, 10]))