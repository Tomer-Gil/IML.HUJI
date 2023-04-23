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

def test_left_join(X: pd.DataFrame, y: Optional[pd.Series] = None):
    non_negative = X[X['labels'] >= 0]
    left_df = X.merge(non_negative, on='id', how='left', indicator=True)
    # temp = X[(X.loc[:, ~X.columns.isin(['date', 'lat', 'long'])] >= 0).all(1)].merge(X.drop_duplicates(), on="id", how='left', indicator=True)
    # temp[temp['_merge'] == "both"]

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
    if y is not None:
        X = X.assign(price=y)

    # Drop rows with empty cells and then remove duplicates
    X.dropna(inplace=True)

    # Another option - replace some empty values with mean
    # for column in X.loc[:, ~X.columns.isin(['date', 'zip', 'lat', 'long'])]:
    #     X[column].replace(np.nan, X[column].mean())
    # Second way
    # X.drop((X.loc[:, ~X.columns.isin(['date', 'lat', 'long'])] < 0).all(1).index, inplace=True)
    # Third way
    # X = X[(X.loc[:, ~X.columns.isin(['date', 'lat', 'long'])] >= 0).all(1)]

    # Cleaning wrong format
    X['date'] = pd.to_datetime(X['date'], errors='coerce')
    X.dropna(subset=['date'], inplace=True)

    # Filter wrong data
    # X = X[
    #     (X['price'] >= 0) |
    #     (X['bedrooms'].apply(float.is_integer).all() & X['bedrooms'] >= 0) |
    #     (X['bathrooms'].is_integer & X['bathrooms'] >= 0) |
    #     (X['sqft_living'] >= 0) |
    #     (X['sqft_living'] >= 0) |
    #     (X['floors'].is_integer() & X['floors'] >= 0) |
    #     (X['waterfront'] in {0, 1}) |
    #     (X['view'] in {0, 1})
    #     ]
    # Another option
    # X = X[
    #     (X['price'] > 0) & (X['bedrooms'] >= 0) & (X['bathrooms'] >= 0) & (X['sqft_living'] >= 0) &
    #     (X['sqft_living'] >= 0) & (X['floors'] >= 0) & (X['waterfront'].isin({0, 1})) &
    #     (X['view'].isin({0, 1}))
    # ]
    X = X[
        (X['bedrooms'] >= 0) & (X['bathrooms'] >= 0)
    ]
    X = X[X['floors'] >= 0]
    X = X[
        (X['condition'] > 0) & (X['grade'] > 0)
    ]
    X = X[
        (X['sqft_living'].abs() >= X['sqft_lot'].abs() * 1/1000)
        & (X['sqft_living15'].abs() >= X['sqft_lot15'] * 1/1000)
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
            (X['waterfront'].isin({0, 1})) & (X['view'].isin({0, 1}))
    ]
    X = X[
        X['price'] > 0
    ]

    # Remove duplicates
    X.drop_duplicates(inplace=True)

    # Handle dummy variables
    X = pd.get_dummies(X)
    # X['zipcode'] = X['zipcode'].astype(str)

    if y is not None:
        columns_but_price = list(X.columns)
        columns_but_price.remove('price')
        return X[columns_but_price], X['price']
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
    X.set_index('id')
    X, y = preprocess_data(X, y)
    pearson_correlation = lambda X, Y: X.cov(Y) / (np.std(X) * np.std(Y))
    corr = {feature: pearson_correlation(X[feature], y) for feature in X.drop(columns=['id', 'date'])}

    corr = pd.DataFrame.from_dict(corr, orient="index", columns=["correlation"])
    corr.sort_values(by="correlation", inplace=True, ascending=False)

    print("feature-price correlation in descending order:")
    print("----------")
    # for feature_corr in corr:
    #     print("{}-price correlation = \t{}".format(
    #         feature_corr[0],
    #         feature_corr[1]
    #     ))
    print(corr.to_string())

    X = X.assign(price=y)

    # fig = go.Figure([
    #     go.Scatter(x=X["sqft_living"], y=X["price"], mode="markers")
    # ])
    # fig.update_layout(dict(
    #     title_text="Correlation between Square footage of the house and its price.",
    #     xaxis_title="Square Footage of the house",
    #     yaxis_title="Price"
    # ))
    # fig.show()
    # fig2 = go.Figure([
    #     go.Scatter(x=X["sqft_lot"], y=X["price"], mode="markers")
    # ])
    # fig2.update_layout(title_text="Correlation between Square Footage of the lot and the house price.")
    # fig2.update_xaxes(title_text="Square footage of lot")
    # fig2.update_yaxes(title_text="Price")
    # fig2.show()
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=X["sqft_living"], y=X["price"], mode="markers", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=X["sqft_lot"], y=X["price"], mode="markers", showlegend=False), row=1, col=2)
    fig.update_layout(
        title="Correlation between Square footage of the house and its price, and "
              "between Square Footage of the lot and the house price.",
        grid=dict(rows=1, columns=2)
    )
    # fig2.update_layout(title_text="Correlation between Square Footage of the lot and the house price.")
    fig.update_xaxes(title_text="Square Footage of the house", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_xaxes(title_text="Square footage of lot", row=1, col=2)
    fig.update_yaxes(title_text="Price", row=1, col=2)
    fig.show()


def fit_model_increasing_percentages(df: pd.DataFrame):
    for percentage in range(10, 101):
        train_samples, train_responses, test_samples, test_responses = split_train_test(
            df.drop(columns="price"), df['price'], percentage/100
        )
        train_samples, train_responses = preprocess_data(train_samples, train_responses)


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df.drop(columns="price"), df['price'])
    # df.drop('price', axis="columns") equivalent to df.drop(columns="price") but less readable in my opinion

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X = preprocess_data(test_X)
    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
