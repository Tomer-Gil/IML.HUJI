import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import loss_functions


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)
    train_error = [model.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
    test_error = [model.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]

    fig = go.Figure([
            go.Scatter(x=list(range(1, n_learners + 1)), y=train_error, name="Train Error", mode="lines"),
            go.Scatter(x=list(range(1, n_learners + 1)), y=test_error, name="Test Error", mode="lines")
        ], layout=go.Layout(
            title="Train- and test errors as function of number of fitted learners",
            xaxis_title="Number of fitted learners",
            yaxis_title="Error rate - [0, 1]"
        )
    )
    fig.write_image("q1_error_as_function_of_num_of_learners_noise_{}.png".format(noise), width=1000, height=700)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=1, cols=4, subplot_titles=["{} Classifiers".format(t) for t in T])
    for i, t in enumerate(T):
        fig.add_traces(
            [
                decision_surface(lambda X: model.partial_predict(X, t), lims[0], lims[1], density=60, showscale=False),
                go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                           marker=dict(
                               color=test_y, symbol=np.where(test_y == 1, "circle", "x")
                           ))
            ],
            rows=1,
            cols=i+1
        )
    fig.update_layout(width=1000, height=700)
    fig.write_image("q2_adaboost_no_noise_decision_boundaries_noise_{}.png".format(noise))

    # Question 3: Decision surface of best performing ensemble
    best_size = np.argmin(np.array([
        model.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)
    ])) + 1
    fig = go.Figure(
        [
            decision_surface(lambda X: model.partial_predict(X, best_size), lims[0], lims[1], density=60, showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                       marker=dict(
                           color=test_y, symbol=np.where(test_y == 1, "circle", "x")
                       ))
        ], layout=go.Layout(
            title=r"Decision surface of ensemble of size {}. Accuracy={}".format(
                best_size, loss_functions.accuracy(test_y, model.partial_predict(test_X, best_size))
            )
        )
    )
    fig.write_image("q3_lowest_test_error_noise_{}.png".format(noise))

    # Question 4: Decision surface with weighted samples
    last_distribution = (model.D_ / np.max(model.D_)) * 20
    fig = go.Figure([
        decision_surface(model.predict, lims[0], lims[1], density=60, showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(
                       size=last_distribution, color=train_y, symbol=np.where(train_y == 1, "circle", "x")
                   ))
    ], layout=go.Layout(
        title="Training Set Samples with Size Proportional to the Last Distribution Used by the AdaBoost algorithm",
        width=1000,
        height=700
    ))
    fig.write_image("q4_with_noise_{}.png".format(noise))


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

