from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    # train_X, train_y, test_X, test_y = split_train_test(X, y, n_samples/len(X))
    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # lambdas_range = np.linspace(1, 500, n_evaluations)
    ridge_lambdas_range = np.linspace(10**-5, 0.7, n_evaluations)
    lasso_lambdas_range = np.linspace(10**-2, 2, n_evaluations)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Ridge", "Lasso"], horizontal_spacing=0.25)
    ridge_scores, lasso_scores = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))
    for i, (ridge_lam, lasso_lam) in enumerate(zip(*[ridge_lambdas_range, lasso_lambdas_range])):
        ridge_scores[i] = cross_validate(RidgeRegression(ridge_lam), train_X, train_y, mean_square_error)
        lasso_scores[i] = cross_validate(Lasso(lasso_lam), train_X, train_y, mean_square_error)
    fig.add_traces([
        go.Scatter(x=ridge_lambdas_range, y=ridge_scores[:, 0], mode="lines", name="Ridge Train Error"),
        go.Scatter(x=ridge_lambdas_range, y=ridge_scores[:, 1], mode="lines", name="Ridge Validation Error"),
        go.Scatter(x=lasso_lambdas_range, y=lasso_scores[:, 0], mode="lines", name="Lasso Train Error"),
        go.Scatter(x=lasso_lambdas_range, y=lasso_scores[:, 1], mode="lines", name="Lasso Validation Error")
    ], rows=[1, 1, 1, 1], cols=[1, 1, 2, 2])
    fig.update_xaxes(title=r"$\lambda \text{ value (Regularization parameter)}$")
    fig.update_yaxes(title="Error")
    fig.update_layout(title="Train and Validation Errors of Ridge and Lasso under 5-Fold Cross-Validation ",
                      width=700,
                      height=350)
    fig.write_image("q2_ridge_{}_to_{}_and_lasso_{}_to_{}.png".format(ridge_lambdas_range.min(),
                                                                      ridge_lambdas_range.max(),
                                                                      lasso_lambdas_range.min(),
                                                                      lasso_lambdas_range.max()))

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = ridge_lambdas_range[np.argmin(ridge_scores[:, 1])]
    best_lam_lasso = lasso_lambdas_range[np.argmin(lasso_scores[:, 1])]
    ridge_loss = RidgeRegression(best_lam_ridge).fit(train_X, train_y).loss(test_X, test_y)
    lasso_loss = mean_square_error(test_y, Lasso(best_lam_lasso).fit(train_X, train_y).predict(test_X))
    least_squares_loss = LinearRegression().fit(train_X, train_y).loss(test_X, test_y)
    print("Best lambda value for Ridge Regression: {}\n"
          "Best lambda value for Lasso Regressions: {}\n"
          "Ridge regressor test error: {}\n"
          "Lasso regressor test error: {}\n"
          "Least-Squares regressor test error: {}".format(
        best_lam_ridge, best_lam_lasso, ridge_loss, lasso_loss, least_squares_loss
    ))


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
