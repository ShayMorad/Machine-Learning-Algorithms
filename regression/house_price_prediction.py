import os
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import numpy as np
import pandas as pd
from typing import NoReturn

averages = dict()

def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # drop insignificant columns
    X = X.drop(["id", "date", "lat", "long"], axis=1)

    # # not dropping "id" for checks with the output csv check
    # X = X.drop(["date", "lat", "long"], axis=1)

    # drop results with price <= 0 as it's an invalid value, and non-existing value, and set X to have same samples as y
    y = y.dropna()
    y = y[y > 0]
    X = X.loc[y.index]

    # fill samples from X with no values and set y to have same samples as X
    X = X.fillna(0)
    y = y.loc[X.index]

    # apply the filter conditions
    X = X[X["waterfront"].isin([0, 1])]
    X = X[X["floors"] > 0]
    X = X[X["yr_built"] >= 1900]
    X = X[(X["yr_built"] <= X["yr_renovated"]) | (X["yr_renovated"] == 0)]
    X = X[X["bathrooms"] > 0]
    X = X[X["bedrooms"] > 0]
    X = X[X["sqft_living"] > 0]
    X = X[X["sqft_lot"] > 0]
    X = X[X["sqft_living15"] > 0]
    X = X[X["sqft_lot15"] > 0]
    X = X[X["sqft_basement"] >= 0]
    X = X[X["sqft_above"] >= 0]
    X = X[X["view"].isin([0, 1, 2, 3, 4])]
    X = X[X["condition"].isin([1, 2, 3, 4, 5])]
    X = X[X["grade"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])]
    y = y.loc[X.index]

    # apply filtering to categorical variables by the logic learnt in rec 3
    X["decade"] = (X["yr_built"] // 10) * 10
    one_hot_encoded = pd.get_dummies(X["decade"], prefix="decade", dtype=int)
    X = pd.concat([X, one_hot_encoded], axis=1)

    # get averages for future use in preprocess_test
    averages["yr_built"] = np.mean([X["yr_built"]])
    averages["sqft_living"] = np.mean([X["sqft_living"]])
    averages["sqft_lot"] = np.mean([X["sqft_lot"]])
    averages["sqft_living15"] = np.mean([X["sqft_living15"]])
    averages["sqft_lot15"] = np.mean([X["sqft_lot15"]])
    averages["bedrooms"] = np.mean([X["bedrooms"]])
    averages["bathrooms"] = np.mean([X["bathrooms"]])
    averages["floors"] = np.mean([X["floors"]])

    X.drop(["yr_built"], axis=1, inplace=True)

    return X, y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # Drop insignificant columns
    X = X.drop(["id", "date", "lat", "long"], axis=1)

    # Fill missing values
    X["sqft_living"].fillna(averages['sqft_living'], inplace=True)
    X["yr_built"].fillna(averages["yr_built"], inplace=True)
    X["sqft_living"].fillna(averages["sqft_living"], inplace=True)
    X["sqft_lot"].fillna(averages["sqft_lot"], inplace=True)
    X["sqft_living15"].fillna(averages["sqft_living15"], inplace=True)
    X["sqft_lot15"].fillna(averages["sqft_lot15"], inplace=True)
    X["bedrooms"].fillna(averages["bedrooms"], inplace=True)
    X["bathrooms"].fillna(averages["bathrooms"], inplace=True)
    X["floors"].fillna(averages["floors"], inplace=True)
    X = X.fillna(value=0)

    # Bound values to right values
    X["waterfront"] = X["waterfront"].clip(lower=0, upper=1)
    X["floors"] = X["floors"].clip(lower=1)
    X["yr_built"] = X["yr_built"].clip(lower=1900)
    X["yr_renovated"] = np.where(X["yr_renovated"] < X["yr_built"], 0, X["yr_renovated"])
    X["bathrooms"] = X["bathrooms"].clip(lower=1)
    X["bedrooms"] = X["bedrooms"].clip(lower=1)
    X["sqft_living"] = X["sqft_living"].clip(lower=1)
    X["sqft_lot"] = X["sqft_lot"].clip(lower=1)
    X["sqft_living15"] = X["sqft_living15"].clip(lower=1)
    X["sqft_lot15"] = X["sqft_lot15"].clip(lower=1)
    X["sqft_basement"] = X["sqft_basement"].clip(lower=0)
    X["sqft_above"] = X["sqft_above"].clip(lower=0)
    X["view"] = X["view"].clip(lower=0, upper=4)
    X["condition"] = X["condition"].clip(lower=1, upper=5)
    X["grade"] = X["grade"].clip(lower=1, upper=13)

    X["decade"] = (X["yr_built"] // 10) * 10
    one_hot_encoded = pd.get_dummies(X["decade"], prefix="decade", dtype=int)
    X = pd.concat([X, one_hot_encoded], axis=1)
    X.drop(["yr_built"], axis=1, inplace=True)

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
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    for col in X.columns:
        # Compute Pearson Correlation
        feature = X[col]
        covariance = np.cov(feature, y)[0, 1]
        std_dev_feature = np.std(feature)
        std_dev_y = np.std(y)
        pearson_corr = covariance / (std_dev_feature * std_dev_y)

        # Create scatter plot
        plt.figure(figsize=(6, 4))
        plt.scatter(feature, y, alpha=0.5)
        plt.title("Feature: {}\nPearson Correlation: {:.2f}".format(col, pearson_corr))
        plt.xlabel(col)
        plt.ylabel("Respone")

        # Save plot
        path = os.path.join(output_path, "{}.png".format(col))
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    random_seed = 69
    test_ratio = 0.25

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Shuffle the indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Calculate split index
    test_size = int(len(X) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Split the data
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train)

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    linear_regression_model = LinearRegression(True)
    # drop missing prices or prices that are not positive, and make sure the test features have same remaining indexes
    y_test = y_test.dropna()
    y_test = y_test[y_test > 0]
    X_test = X_test.loc[y_test.index]

    percentages = np.arange(10, 101)
    losses = np.zeros((91, 10))

    for index, percentage in enumerate(percentages):
        for j in range(10):
            # Sample p% of the overall training data
            X_train_sample = X_train.sample(frac=(percentage / 100), random_state=j)
            y_train_sample = y_train.loc[X_train_sample.index]

            # convert DataFrame to NumPy array
            X_train_sample_np = X_train_sample.to_numpy()
            y_train_sample_np = y_train_sample.to_numpy()

            # fit the model
            linear_regression_model.fit(X_train_sample_np, y_train_sample_np)

            # calculate loss
            X_test_np = X_test.to_numpy()
            y_test_np = y_test.to_numpy()

            losses[index, j] = linear_regression_model.loss(X_test_np, y_test_np)

    # calculate mean loss and std loss
    mean_losses = losses.mean(axis=1)
    std_losses = losses.std(axis=1)

    # create plot
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, mean_losses, label='Mean Loss')
    plt.fill_between(percentages, mean_losses - 2 * std_losses, mean_losses + 2 * std_losses, alpha=0.2,
                     label='Confidence Interval')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Mean Squared Error Loss')
    plt.title('Loss vs. Training Data Size')
    plt.legend()
    plt.savefig("loss_vs_training.png")
    plt.close()
