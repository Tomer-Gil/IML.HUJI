from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import plotly.express as px


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be a
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    traces = []
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/{}".format(f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        p = Perceptron(callback=lambda: losses.append(p.loss(X, y)))
        p.fit(X, y)
        # traces.append(px.line(x=list(range(len(losses))), y=losses))
        traces.append(go.Scatter(x=list(range(len(losses))), y=losses, mode="lines"))

    # Plot figure of loss as function of fitting iteration
    fig = make_subplots(1, 2, subplot_titles=(r"Linearly Separable", r"Linearly Inseparable"))
    fig.add_traces(traces, rows=[1, 1], cols=[1, 2])
    fig.update_layout(title=r"Misclassification Error as Function of Fitting Iteration", showlegend=False)
    fig.update_xaxes(title_text="Fitting Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Misclassification Error Rate", row=1, col=1)
    fig.update_xaxes(title_text="Fitting Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Misclassification Error Rate", row=1, col=2)
    fig.write_image("misclassification_error_by_fitting_iteration.png", width=1000, height=700)


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/{}".format(f))

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        lda = LDA()
        lda.fit(X, y)
        gnb_pred = gnb.predict(X)
        lda_pred = lda.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            r"{} (Accuracy = {:.2%})".format(name, accuracy(y, pred))
            for name, pred in zip(["Gaussian Naive Bayes", "LDA"], [gnb_pred, lda_pred])
        ], horizontal_spacing=0.01, vertical_spacing=0.03)

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(
                color=gnb.predict(X), symbol=class_symbols[y.astype(int)], colorscale=class_colors(3)
            )
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(
                color=lda.predict(X), symbol=class_symbols[y.astype(int)], colorscale=class_colors(3)
            )
        ), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(
                x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers", marker=dict(symbol="x", color="black", size=15)
            ),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers", marker=dict(symbol="x", color="black", size=15)
            ),
            row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(gnb.classes_)):
            fig.add_trace(get_ellipse(gnb.mu_[i], gnb.vars_[i]), row=1, col=1)
        for i in range(len(lda.classes_)):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        fig.update_layout(
            title=r"Predictions of Gaussian Naive Bayes and LDA classifiers over {} dataset.".format(f),
            showlegend=False)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.write_image("compare_lda_and_naive_with_{}.png".format(f), width=1000)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
