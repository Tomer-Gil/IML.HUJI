import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def sanitize_data(df: pd.DataFrame) -> None:
    pass


def load_data(filename: str) -> pd.DataFrame:
    """
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
    # df.dropna().drop_duplicates()
    df.rename(columns={'Temp': 'Temp_Celsius'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    # df.dropna(inplace=True)
    # Filter of year - to be updated each year
    # -273.15 - absolute zero temperature (in Celsius). I choose 100 degrees as a temperature symbolizes impossible
    # temperature to be measured on earth, at least where human beings are living
    df = df.loc[
        df['Year'].between(0, 2023)
        & df['Month'].between(1, 12)
        & df['Day'].between(1, 31)
        & df['Temp_Celsius'].between(-20, 100)
    ]
    # df = df[(df['Year']>0) & (df['Month'] > 0) & (df['Month'] < 13) & (df['Day']> 0) & (df['Day'] <32) & (df['Temp_Celsius'] > -20)]

    # Validate the date represented by the Year, Month, Day's columns match the date represented in the Date column
    df['date_from_year_month_day'] = pd.to_datetime(
        dict(
            year=df['Year'],
            month=df['Month'],
            day=df['Day']
        )
    )
    df = df.loc[df['Date'] == df['date_from_year_month_day']]
    df.drop(columns='date_from_year_month_day', inplace=True)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


def day_of_year_to_temperature_relation(df: pd.DataFrame) -> None:
    df = df.loc[df['Country'] == "Israel"]
    df['Year'] = df['Year'].astype(str)
    df.sort_values(by='Date', axis='rows', inplace=True)
    fig = px.scatter(df, x='DayOfYear', y='Temp_Celsius', color='Year', title='Average Temperature by Day of Year',
                     labels={
                         'DayOfYear': "Day of Year",
                         'Temp_Celsius': "Average Temperature (Celsius)"
                     })
    # To-Do - remove extremely low Temperatures
    # fig.update_layout(yaxis_range=[-10, 50])
    fig.show()
    # fig = px.line(df.sort_values(by='Date', axis='rows'), x='Date', t='Temp_Celsius')
    # fig.update_xaxes()


def std_by_month(df: pd.DataFrame) -> None:
    res = df.groupby('Month')['Temp_Celsius'].std().reset_index()
    # Equivalent wat - df.groupby('Month')['Temp_Celsius'].agg("std").reset_index()
    px.bar(res, x='Month', y='Temp_Celsius', title="Standard deviation of average daily temperature in Israel"
                                                   " by months",
           labels={
               'Temp_Celsius': 'Temperature (Celsius)'
           }).show()


def q3(df: pd.DataFrame) -> None:
    df2 = df.groupby(['Country', 'Month'])['Temp_Celsius'].agg(["std", "mean"]).reset_index()
    px.line(df2, x='Month', y='mean', color='Country', error_y='std',
            title="Monthly average temperatures by country, with standard deviation as error bar",
            labels={
                "mean": r"$\text{Mean (} \pm \text{ standard deviation)}$"
            }).show()


def q3_another_way(df: pd.DataFrame) -> None:
    df3 = df.groupby(['Country', 'Month'], as_index=False).agg(std=("Temp_Celsius", "std"),
                                                              mean=("Temp_Celsius", "mean")).reset_index()
    px.line(df3, x='Month', y='mean', color='Country', error_y='std',
            title="Monthly average temperatures by country, with standard deviation as error bar",
            labels={
                "mean": r"$\text{Mean (} \pm \text{ standard deviation)}$"
            }
            ).show()


def q4(df: pd.DataFrame) -> None:
    train_samples, train_responses, test_samples, test_responses = split_train_test(
        df.drop(columns='Temp_Celsius'), df['Temp_Celsius']
    )
    loss_per_degree = []
    for k in range(1, 11):
        est = PolynomialFitting(k)
        est.fit(train_samples['Month'], train_responses)
        loss_per_degree.append(round(est.loss(test_samples['Month'], test_responses), 2))
    print("Loss per polynom degree:\n{}".format(np.array(loss_per_degree)))
    px.bar(x=list(range(1, 11)), y=loss_per_degree, title="Loss By Polynomial Rank",
           labels={
               "x": "Rank",
               "y": "Loss"
           }).show()


def q5(df: pd.DataFrame) -> None:
    israel_df = df.loc[df['Country'] == "Israel"]
    est = PolynomialFitting(1)
    est.fit(israel_df.drop(columns="Temp_Celsius")['Month'], israel_df['Temp_Celsius'])
    countries_loss = dict()
    countries = list(df['Country'].unique())
    countries.remove("Israel")
    for country in countries:
        country_df = df.loc[df['Country'] == country]
        countries_loss[country] = est.loss(country_df.drop(columns='Temp_Celsius')['Month'], country_df['Temp_Celsius'])
    px.bar(x=countries_loss.keys(), y=countries_loss.values(),
           title="Loss of a model fitted over Israel over different countries",
           labels={
               "x": "Country",
               "y": "Loss (expressing the difference between model prediction and the real responses)"
           }).show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    # day_of_year_to_temperature_relation(df)
    # std_by_month(df)

    # Question 3 - Exploring differences between countries
    q3(df)

    # Question 4 - Fitting model for different values of `k`
    # q4(df.loc[df['Country'] == "Israel"])

    # Question 5 - Evaluating fitted model on different countries
    # q5(df)
