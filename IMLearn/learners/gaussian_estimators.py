from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None
        self.samples_num_ = None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.samples_num_ = X.size

        self.mu_ = np.mean(X)
        self.var_ = (np.nansum((X - self.mu_) ** 0.5) / self.samples_num_) if self.biased_ \
            else (np.nansum((X - self.mu_) ** 0.5) / (self.samples_num_ - 1))

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        pdf_coefficient = 1 / (2 * np.pi * self.var_ ** 2) ** 0.5
        pdf_inner_exp = lambda sample: (-1 / (2 * self.var_ ** 2)) * (sample - self.mu_) ** 2


        return np.array([pdf_coefficient * np.exp(pdf_inner_exp(sample)) for sample in X])

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        return (-1 / 2) * np.sum(np.log(2 * np.pi * sigma ** 2) + ((sample - mu) / sigma) ** 2 for sample in X)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False
        self.samples_num_, self.features_num_, self.det_ = None, None, None # Num of cols, i.e. features

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # Num of rows, i.e. samples and num of cols, i.e. features
        self.samples_num_, self.features_num_ = X.shape[0], X.shape[1]

        self.mu_ = np.mean(X, axis=0)

        self.cov_ = (1 / (self.samples_num_ - 1)) * (np.transpose(X - self.mu_) @ (X - self.mu_))
        self.det_ = np.linalg.det(self.cov_)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        pdf_coefficient = 1 / ((2 * np.pi) ** self.features_num_ * np.linalg.det(self.cov_)) ** 0.5
        pdf_inner_exp = lambda sample: (-1 / 2) * np.mathmul(
            np.mathmul(np.subtract(sample, self.mu_), np.linalg.inv(self.cov_)),
            np.subtract(sample, self.mu_)
        )

        return np.ndarray(pdf_coefficient * np.exp(pdf_inner_exp(sample)) for sample in X)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        samples_num, features_num = X.shape
        det = np.linalg.det(cov)
        triple_matrix_prod = np.einsum("bi,ij,bj", X - mu, np.linalg.inv(cov), X - mu)
        return -0.5 * (features_num * samples_num * np.log(2 * np.pi)
                       + samples_num * np.log(det)
                       + triple_matrix_prod)
            # - 0.5 * np.sum(
            #     (sample - mu).dot(np.linalg.inv(cov) @ (sample - mu)) for sample in X
            # )


if __name__ == "__main__":
    mean = [1, 2, 0, 4]
    cov = [
        [1, -1, 2, 0],
        [-1, 4, -1, 1],
        [2, -1, 6, -2],
        [0, 1, -2, 4]
    ]
    X = np.random.multivariate_normal(mean, cov, 6)

    est = MultivariateGaussian()
    est.fit(X)
    print("done")
