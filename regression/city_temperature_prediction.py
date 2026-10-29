import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from polynomial_fitting import PolynomialFitting

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
    # load the dataset
    X = pd.read_csv(filename, parse_dates=['Date'])

    # handle invalid data, removing samples with no data and samples with temperature below -20
    X.dropna(inplace=True)
    X = X[X["Temp"] > -20]

    # Add 'DayOfYear' column
    X['DayOfYear'] = X['Date'].dt.dayofyear

    return X


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_data = df[df['Country'] == 'Israel']

    years = israel_data["Year"].unique()
    for year in years:
        year_data = israel_data[israel_data['Year'] == year]
        plt.scatter(year_data["DayOfYear"], year_data["Temp"], label=str(year))
    plt.title('Average Daily Temperature Change Over Day of Year (Israel)')
    plt.xlabel('Day of Year')
    plt.ylabel('Average Daily Temperature')
    plt.legend(title='Year', loc='upper left')
    plt.savefig("average_daily_temperatures_change_by_dayofyear.png")
    plt.close()

    # group by month and calculate standard deviation of daily temperatures
    monthly_std = israel_data.groupby('Month')['Temp'].std()

    # Bar plot: Standard deviation of daily temperatures by month
    monthly_std.plot(kind='bar')
    plt.title('Standard Deviation of Daily Temperatures by Month')
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation of Daily Temperature')
    plt.xticks(rotation=1)
    plt.savefig("std_daily_temperatures_by_month.png")
    plt.close()

    # Question 4 - Exploring differences between countries
    # group the samples according to 'Country' and 'Month' and calculate average and standard deviation of temperature
    country_monthly_stats = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()

    # Plot line plot with error bars (standard deviation) color coded by country
    for country in country_monthly_stats['Country'].unique():
        country_data = country_monthly_stats[country_monthly_stats['Country'] == country]
        plt.errorbar(country_data['Month'], country_data['mean'], yerr=country_data['std'], label=country)

    plt.title('Average Monthly Temperature by Country')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature')
    plt.legend(title='Country', loc='upper left')
    plt.savefig("average_temperatures_by_country.png")
    plt.close()

    # Question 5 - Fitting model for different values of `k`
    temp = israel_data["Temp"]
    dayofyear = israel_data["DayOfYear"]
    random_seed = 69
    test_ratio = 0.25

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Shuffle the indices
    indices = np.arange(len(dayofyear))
    np.random.shuffle(indices)

    # Calculate split index
    test_size = int(len(dayofyear) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Split the data
    X_train, X_test = dayofyear.iloc[train_indices], dayofyear.iloc[test_indices]
    y_train, y_test = temp.iloc[train_indices], temp.iloc[test_indices]
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    losses = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        loss = model.loss(X_test, y_test)
        loss_rounded = round(loss, 2)
        losses.append(loss_rounded)
    for degree, error in enumerate(losses, start=1):
        print("degree k={}, error={}".format(degree, error))

    plt.bar(range(1, 11), losses, tick_label=range(1, 11))
    plt.title('Test Error for Different Values of k')
    plt.xlabel('Degree of Polynomial (k)')
    plt.ylabel('Test Error')
    plt.savefig("errors_diff_degrees.png")
    plt.close()

    # Question 6 - Evaluating fitted model on different countries
    X = dayofyear.to_numpy()
    y = temp.to_numpy()
    model = PolynomialFitting(5)
    model.fit(X, y)
    errors_by_country = {}
    countries = [country for country in df["Country"].unique() if country != "Israel"]
    for country in countries:
        country_data = df[df['Country'] == country]
        X_country = country_data['DayOfYear'].to_numpy()
        y_country = country_data['Temp'].to_numpy()

        error_country = model.loss(X_country, y_country)

        errors_by_country[country] = error_country

    plt.bar(errors_by_country.keys(), errors_by_country.values())
    plt.title('Model Error over Other Countries')
    plt.xlabel('Country')
    plt.ylabel('Model Error')
    plt.xticks(rotation=1)
    plt.savefig("errors_other_countries_with_israel.png")
    plt.close()
