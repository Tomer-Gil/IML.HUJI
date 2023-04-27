from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from plotly.subplots import make_subplots


def preprocess_data_for_train(X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.Series):
    X.dropna(subset=['date'], inplace=True)

    X = X[
        (X['bedrooms'] >= 0) & (X['bathrooms'] >= 0)
        ]
    X = X[X['floors'] >= 0]
    X = X[
        (X['condition'] > 0) & (X['grade'] > 0)
        ]
    X = X[
        (X['sqft_living'].abs() >= X['sqft_lot'].abs() * 1 / 1000)
        & (X['sqft_living15'].abs() >= X['sqft_lot15'] * 1 / 1000)
        ]
    X[['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']] = X[
        ['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']
    ].abs()
    X = X[
        (X['sqft_above'] >= 0) & (X['sqft_basement'] >= 0)
        ]
    # X = X.loc[
    #     X['yr_built'].str.isdigit(), X['yr_renovated'].str.isdigit()
    # ]
    X = X.loc[
        ~((X['yr_renovated'] < X['yr_built']) & (X['yr_renovated'] != 0))
    ]
    X = X[
        (X['waterfront'].isin({0, 1})) & (X['view'].isin({0, 1, 2, 3, 4}))
        ]
    y = y[y > 0]

    # Handle dummy variables
    X['zipcode'] = X['zipcode'].astype(str)
    X = pd.get_dummies(X, columns=['zipcode'])
    return X, y


def preprocess_data_for_test(test_samples: pd.DataFrame) -> pd.DataFrame:
    test_samples.loc[pd.isnull(test_samples['date']), 'date'] = test_samples['date'].fillna(
        test_samples['date'].mean()
    )
    test_samples.loc[test_samples['bedrooms'] < 0]['bedrooms'] =\
        test_samples.loc[test_samples['bedrooms'] >= 0]['bedrooms'].mean()
    test_samples.loc[test_samples['bathrooms'] < 0]['bathrooms'] = \
        test_samples.loc[test_samples['bathrooms'] >= 0]['bathrooms'].mean()
    test_samples.loc[~(test_samples['floors'] >= 0)]['floors'] = \
        test_samples.loc[test_samples['floors'] >= 0]['floors'].mean()
    test_samples.loc[~(test_samples['condition'].isin(range(1, 6))), 'condition'] = \
        test_samples.loc[test_samples['condition'].isin(range(1, 6))].mode()['condition'].median()
    test_samples.loc[~(test_samples['grade'].isin(range(1, 14))), 'grade'] = \
        test_samples.loc[test_samples['grade'].isin(range(1, 14))]['grade'].mode().median()
    test_samples.loc[~(test_samples['sqft_living'] > 0), 'sqft_living'] = \
        test_samples.loc[test_samples['sqft_living'] > 0]['sqft_living'].mean()
    test_samples.loc[~(test_samples['sqft_living15'] > 0), 'sqft_living15'] = \
        test_samples.loc[test_samples['sqft_living15'] > 0]['sqft_living15'].mean()
    test_samples.loc[~(test_samples['sqft_lot'] > 0), 'sqft_lot'] = \
        test_samples.loc[test_samples['sqft_lot'] > 0]['sqft_lot'].mean()
    test_samples.loc[~(test_samples['sqft_lot15'] > 0), 'sqft_lot15'] = \
        test_samples.loc[test_samples['sqft_lot15'] > 0]['sqft_lot15'].mean()
    test_samples.loc[~(test_samples['sqft_above'] >= 0), 'sqft_above'] = \
        test_samples.loc[test_samples['sqft_above'] >= 0]['sqft_above'].mean()
    test_samples.loc[~(test_samples['sqft_basement'] >= 0), 'sqft_basement'] = \
        test_samples.loc[test_samples['sqft_basement'] >= 0]['sqft_basement'].mean()
    test_samples.loc[(test_samples['yr_renovated'] < test_samples['yr_built']) & (test_samples['yr_renovated'] != 0)] = \
        test_samples[
            ~((test_samples['yr_renovated'] < test_samples['yr_built']) & (test_samples['yr_renovated'] != 0))
        ]['yr_renovated'].median()
    test_samples.loc[~(test_samples['waterfront'].isin({0, 1}))]['waterfront'] = \
        test_samples.loc[test_samples['waterfront'].isin({0, 1})]['waterfront'].mode().median()
    test_samples.loc[~(test_samples['view'].isin(range(5))), 'view'] = \
        test_samples.loc[test_samples['view'].isin(range(5))]['view'].mode().median()
    return test_samples


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X.drop(columns='id', inplace=True)
    if y is not None:
        # Drop rows with empty cells and then remove duplicates
        X = X.dropna().drop_duplicates()

    # Cleaning wrong format
    X['date'] = pd.to_datetime(X['date'], errors='coerce')
    X.loc[~pd.isnull(X['date']), 'date'] = X.loc[~pd.isnull(X['date']), 'date'].values.astype(np.int64) / 10 ** 9
    if y is not None:
        X, y = preprocess_data_for_train(X, y)
    else:
        X = preprocess_data_for_test(X)
    X['date'] = X['date'].astype(np.float64)
    if y is not None:
        return X.loc[X.index.intersection(y.index)], y.loc[X.index.intersection(y.index)]
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # X, y = preprocess_data(X, y)
    pearson_correlation = lambda X, Y: X.cov(Y) / (np.std(X) * np.std(Y))
    corr = {feature: pearson_correlation(X[feature], y) for feature in X}

    corr = pd.DataFrame.from_dict(corr, orient="index", columns=["correlation"])
    corr.sort_values(by="correlation", inplace=True, ascending=False)

    print("feature-price correlation in descending order:")
    print("----------")
    print(corr.to_string())

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=X["sqft_living"], y=y, mode="markers", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=X["sqft_lot"], y=y, mode="markers", showlegend=False), row=1, col=2)
    fig.update_layout(
        title="Correlation between Square footage of the house and its price, and <br>"
              "between Square Footage of the lot and the house price.",
        grid=dict(rows=1, columns=2)
    )
    fig.update_xaxes(title_text="Square Footage of the house", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_xaxes(title_text="Square footage of lot", row=1, col=2)
    fig.update_yaxes(title_text="Price", row=1, col=2)
    fig.write_image(output_path, width=1000, height=700)


def fit_model_increasing_percentages(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
                                     test_y: pd.Series) -> None:
    p_loss_map = np.zeros((91, 10))
    for i, percentage in enumerate(range(10, 101)):
        for j in range(10):
            _fit_X = train_X.sample(frac=percentage/100)
            _fit_y = train_y.loc[_fit_X.index]
            est = LinearRegression()
            est.fit(_fit_X, _fit_y)
            p_loss_map[i, j] = est.loss(test_X, test_y)
    fig = go.Figure(data=[
        go.Scatter(x=list(range(10, 101)), y=p_loss_map.mean(axis=1), mode="markers+lines", name="Mean Prediction"),
        go.Scatter(x=list(range(10, 101)), y=p_loss_map.mean(axis=1) - 2 * p_loss_map.std(axis=1), fill=None,
                   mode="lines", line=dict(color="lightgray"), showlegend=False),
        go.Scatter(x=list(range(10, 101)), y=p_loss_map.mean(axis=1) + 2 * p_loss_map.std(axis=1), fill="tonexty",
                   mode="lines", line=dict(color="lightgray"), showlegend=False)
    ])
    fig.update_layout(
        title="House price prediction mean by relative size of training set of the given dataset",
        xaxis_title="Relative size of training set out of dataset (percentages)",
        yaxis_title="House price prediction mean"
    )
    fig.write_image("house_price_prediction_mse_over_test_set.png", width=1000, height=700)


def handle_get_dummies_after_preprocess(train_columns: pd.Index, test_samples: pd.DataFrame) -> pd.DataFrame:
    test_samples = pd.get_dummies(test_samples, columns=['zipcode'])
    zipcodes_to_remove = test_samples.columns.difference(train_columns)
    test_samples[train_columns.difference(test_samples.columns)] = 0
    return test_samples[test_samples.columns.difference(zipcodes_to_remove)][train_columns]


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df.drop(columns="price"), df['price'])
    train_y.dropna(inplace=True)
    test_y.dropna(inplace=True)
    train_X = train_X.loc[train_y.index]
    test_X = test_X.loc[test_y.index]

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X = preprocess_data(test_X)
    test_X = handle_get_dummies_after_preprocess(train_X.columns, test_X)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, "features_correlation.png")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit_model_increasing_percentages(train_X, train_y, test_X, test_y)
