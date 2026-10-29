import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)

    # Randomly choose 50 unique samples for training (without replacement)
    n_total = len(X)
    train_idx = np.random.choice(n_total, size=n_samples, replace=False)

    # Get test indices as the set difference
    all_idx = np.arange(n_total)
    test_idx = np.setdiff1d(all_idx, train_idx)

    # Create train and test splits
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    # Define meaningful lam ranges as instructed in the exercise
    ridge_lambdas = np.linspace(1e-7, 0.4, n_evaluations)
    lasso_lambdas = np.linspace(1e-6, 1.0, n_evaluations)

    ridge_errors = np.zeros((n_evaluations, 2))  # [train_error, val_error]
    lasso_errors = np.zeros((n_evaluations, 2))

    for i, lam in enumerate(ridge_lambdas):
        ridge_model = RidgeRegression(lam=lam, include_intercept=True)
        ridge_errors[i] = cross_validate(ridge_model, X_train, y_train, cv=5)

    for i, lam in enumerate(lasso_lambdas):
        lasso_model = Lasso(alpha=lam, include_intercept=True)
        lasso_errors[i] = cross_validate(lasso_model, X_train, y_train, cv=5)

    # Plotting
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Ridge Regression", "Lasso Regression"])

    # Ridge
    fig.add_trace(go.Scatter(x=ridge_lambdas, y=ridge_errors[:, 0],
                             name="Train Error (Ridge)", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ridge_lambdas, y=ridge_errors[:, 1],
                             name="Validation Error (Ridge)", line=dict(color="red")), row=1, col=1)

    # Lasso
    fig.add_trace(go.Scatter(x=lasso_lambdas, y=lasso_errors[:, 0],
                             name="Train Error (Lasso)", line=dict(color="blue", dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=lasso_lambdas, y=lasso_errors[:, 1],
                             name="Validation Error (Lasso)", line=dict(color="red", dash="dash")), row=1, col=2)

    fig.update_layout(title="Train and Validation Error vs λ",
                      xaxis_title="λ (Regularization Strength)",
                      width=1000, height=500)
    fig.update_xaxes(title_text="λ (Regularization Strength)", type="linear", row=1, col=1)
    fig.update_xaxes(title_text="λ (Regularization Strength)", type="linear", row=1, col=2)
    fig.update_yaxes(title_text="MSE")

    fig.write_image("lasso_ridge_cross_validate_errors.png")

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    # --- Find best lambdas from CV ---
    ridge_lambda = ridge_lambdas[np.argmin(ridge_errors[:, 1])]
    lasso_lambda = lasso_lambdas[np.argmin(lasso_errors[:, 1])]

    # --- Train models on full training set (50 samples) ---
    ridge_model = RidgeRegression(lam=ridge_lambda, include_intercept=True)
    ridge_model.fit(X_train, y_train)

    lasso_model = Lasso(alpha=lasso_lambda, include_intercept=True)
    lasso_model.fit(X_train, y_train)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # --- Print best lambdas ---
    print(f"Best λ for Ridge: {np.round(ridge_lambda, 10)}")
    print(f"Best λ for Lasso: {np.round(lasso_lambda, 10)}")

    # --- Evaluate on test set ---
    ridge_test_error = ridge_model.loss(X_test, y_test)
    lasso_test_error = lasso_model.loss(X_test, y_test)
    linear_test_error = linear_model.loss(X_test, y_test)

    print(f"Ridge Test Error: {np.round(ridge_test_error, 10)}")
    print(f"Lasso Test Error: {np.round(lasso_test_error, 10)}")
    print(f"Linear Regression Test Error: {np.round(linear_test_error, 10)}")


if __name__ == '__main__':
    np.random.seed(456)
    select_regularization_parameter()
