from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly import subplots
pio.templates.default = "simple_white"
from scipy.stats import norm


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, var = 10, 1
    X = np.random.normal(mu, var, 1000)
    gaus = UnivariateGaussian()
    gaus.fit(X)
    print((gaus.mu_, gaus.var_))


    # Question 2 - Empirically showing sample mean is consistent
    # dists = [np.abs(UnivariateGaussian().fit(X[:n]).mu_ - mu) for n in range(10, 1020, 10)]
    # fig = subplots.make_subplots(rows=1, cols=1).add_traces(
    #     [go.Scatter(x=np.arange(0, 1010, 10), y=dists, mode="lines+markers")]
    # ).update_layout(
    #     template=pio.templates.default,
    #     title="Distance between estimated and true- value of expectation, as a function of the sample size",
    #     xaxis_title=r"Sample size",
    #     yaxis_title=r"Estimated expectation value"
    # )
    # fig.show()
    #
    # # Question 3 - Plotting Empirical PDF of fitted model
    # go.Figure([
    #     go.Scatter(x=np.sort(X), y=gaus.pdf(np.sort(X)), mode="lines+markers")
    # ]).update_layout(
    #     template=pio.templates.default,
    #     title=r"PDFs of 1,000 samples drawn from the normal distribution with unknown mean and variance,<br />"
    #           r"calculated from a fitted model",
    #     xaxis_title=r"Sorted samples value",
    #     yaxis_title=r"Sample PDF"
    # ).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    Sigma = np.array(
        [[1, 0.2, 0, 0.5],
        [0.2, 2, 0, 0],
        [0, 0, 1, 0],
        [0.5, 0 , 0, 1]]
    )
    X = np.random.multivariate_normal(mu, Sigma, 1000)
    gaus = MultivariateGaussian()
    fitted = gaus.fit(X)
    print("Estimated means vector:\n{}\n\nEstimated covariance matrix:\n{}".format(
        fitted.mu_, fitted.cov_
    ))
    log_values = np.zeros((200, 200))
    axis_values = np.linspace(-10, 10, 200)
    # Question 5 - Likelihood evaluation
    for i, f1 in enumerate(axis_values):
        for j, f3 in enumerate(axis_values):
            log_values[i, j] = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), Sigma, X)

    go.Figure(go.Heatmap(x=axis_values, y=axis_values, z=log_values)).update_layout(
        template=pio.templates.default,
        title=r"Log-likelihood values for different mean vectors, differ from each other by the first (f1)"
              r"and the third(f3) coordinates.<br> I.e., how likely is it for each pair of (f1, f3) to be the actual"
              r"first and third coordinated, correspondingly, of the actual mean vector of the distribution from"
              r"which the samples were drawn.",
        xaxis_title=r"f1 values",
        yaxis_title=r"f3 values"
    ).show()


    # Question 6 - Maximum likelihood
    f1, f3 = np.round(axis_values[list(np.unravel_index(log_values.argmax(), log_values.shape))], 3)
    print("Maximum likelihood is achieved from (f1, f3) pair: ({}, {})".format(f1, f3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
