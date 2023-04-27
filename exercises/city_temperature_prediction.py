import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Q. 1
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.dropna().drop_duplicates()
    df.rename(columns={'Temp': 'Temp_Celsius'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    # Filter of year - to be updated each year
    df = df[
        df['Year'].between(0, 2023)
        & df['Month'].between(1, 12)
        & df['Day'].between(1, 31)
        & df['Temp_Celsius'].between(-20, 100)
    ]
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


def day_of_year_to_temperature_relation(df: pd.DataFrame) -> None:
    """
    Q. 2 a
    Parameters
    ----------
    df

    Returns
    -------

    """
    df = df[df['Country'] == "Israel"]
    df['Year'] = df['Year'].astype(str)
    # df.sort_values(by='Date', axis='rows', inplace=True)
    px.scatter(df, x='DayOfYear', y='Temp_Celsius', color='Year', title='Average Temperature by Day of Year',
                     labels={
                         'DayOfYear': "Day of Year",
                         'Temp_Celsius': "Average Temperature (Celsius)"
                     }).write_image("temp_by_dayOfYear.png", width=1000, height=700)


def std_by_month(df: pd.DataFrame) -> None:
    """
    Q. 2 b
    Parameters
    ----------
    df

    Returns
    -------

    """
    res = df.groupby('Month')['Temp_Celsius'].std().reset_index()
    px.bar(res, x='Month', y='Temp_Celsius', title="Standard deviation of average daily temperature in Israel"
                                                   " by months",
           labels={
               'Temp_Celsius': 'Temperature (Celsius)'
           }).write_image("std_by_month.png", width=1000, height=700)


def q3(df: pd.DataFrame) -> None:
    df2 = df.groupby(['Country', 'Month'])['Temp_Celsius'].agg(["std", "mean"]).reset_index()
    px.line(df2, x='Month', y='mean', color='Country', error_y='std',
            title="Monthly average temperatures by country, with standard deviation as error bar",
            labels={
                "mean": r"$\text{Mean (} \pm \text{ standard deviation)}$"
            }).write_image("monthly_avg_temps_by_country.png", width=1000, height=700)


def q4(df: pd.DataFrame) -> None:
    train_samples, train_responses, test_samples, test_responses = split_train_test(
        df.drop(columns=['Temp_Celsius']), df['Temp_Celsius'])
    loss_per_degree = []
    for k in range(1, 11):
        est = PolynomialFitting(k)
        est.fit(train_samples['DayOfYear'], train_responses)
        loss_per_degree.append(round(est.loss(test_samples['DayOfYear'], test_responses), 2))
    print("Loss per polynom degree:\n{}".format(np.array(loss_per_degree)))
    px.bar(x=list(range(1, 11)), y=loss_per_degree, title="Loss By Polynomial Rank", text_auto='0.3s',
           labels={
               "x": "Rank",
               "y": "Loss"
           }).write_image("loss_per_polynom_degree.png", width=1000, height=700)


def q5(df: pd.DataFrame) -> None:
    israel_df = df.loc[df['Country'] == "Israel"]
    est = PolynomialFitting(5)
    est.fit(israel_df.drop(columns="Temp_Celsius")['DayOfYear'], israel_df['Temp_Celsius'])
    countries_loss = dict()
    countries = list(df['Country'].unique())
    countries.remove("Israel")
    for country in countries:
        country_df = df.loc[df['Country'] == country]
        countries_loss[country] = est.loss(country_df.drop(columns='Temp_Celsius')['DayOfYear'], country_df['Temp_Celsius'])
    px.bar(x=countries_loss.keys(), y=countries_loss.values(),
           title="Loss of a model fitted over Israel over different countries",
           text_auto="0.3s",
           labels={
               "x": "Country",
               "y": "Loss (expressing the difference between model prediction and the real responses)"
           }).write_image("loss_over_countries_besides_israel_fitted_over_israel.png", width=1000, height=700)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    # day_of_year_to_temperature_relation(df)
    # std_by_month(df)

    # Question 3 - Exploring differences between countries
    # q3(df)

    # Question 4 - Fitting model for different values of `k`
    # q4(df.loc[df['Country'] == "Israel"])

    # Question 5 - Evaluating fitted model on different countries
    q5(df)
