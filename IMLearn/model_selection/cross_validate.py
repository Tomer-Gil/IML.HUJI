from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    ids = np.arange(X.shape[0])
    folds = np.array_split(ids, cv)
    train_scores, validation_scores = [], []
    for fold in folds:
        train_mask = ~np.isin(ids, fold)
        model = deepcopy(estimator)
        model.fit(X[train_mask], y[train_mask])
        train_scores.append(scoring(y[train_mask], model.predict(X[train_mask])))
        validation_scores.append(scoring(y[np.isin(ids, fold)], model.predict(X[np.isin(ids, fold)])))
    return np.array(train_scores).mean(), np.array(validation_scores).mean()

