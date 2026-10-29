import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from sklearn.metrics import roc_curve, auc
from cross_validate import *
from loss_functions import misclassification_error


# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    recorded_weights = []
    recorded_vals = []

    def gd_callback(solver, weights, val, grad, t, eta, delta, **kwargs):
        recorded_vals.append(val)
        recorded_weights.append(weights)

    return gd_callback, recorded_vals, recorded_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    empty = np.empty(shape=(0,))

    for eta in etas:
        for module, name in [(L1, "L1"), (L2, "L2")]:
            gd_callback, values, weights = get_gd_state_recorder_callback()

            gd = GradientDescent(learning_rate=FixedLR(eta), callback=gd_callback)
            gd.fit(module(weights=init), X=empty, y=empty)

            fig_descent = plot_descent_path(module=module,
                                            descent_path=np.array(weights),
                                            title=f"Descent Path of {name} Module with Fixed Learning Rate η={eta}")
            fig_descent.write_image(f"./{name}_{eta}_descent.png")

            fig_convergence = go.Figure([go.Scatter(y=values, mode="markers")],
                                        layout=go.Layout(
                                            title=f"Convergence Rate of {name} Module with Fixed Learning Rate η={eta}"))

            fig_convergence.update_layout(width=650, height=500) \
                .update_xaxes(title_text="Iteration") \
                .update_yaxes(title_text=f"Convergence (w {name} norm)")

            fig_convergence.write_image(f"./{name}_{eta}_convergence.png")

            print(
                f"The lowest loss of a {name} module with a fixed learning rate η={eta} = {np.round(np.min(values), 3)}")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Fit logistic regression model
    lr = 1e-4
    max_iter = 20000
    callback, losses, weights = get_gd_state_recorder_callback()
    gd_solver = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter, callback=callback)
    logistic_model = LogisticRegression(solver=gd_solver, include_intercept=True).fit(X_train, y_train)

    # Predict probabilities and compute ROC curve
    train_probabilities = logistic_model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, train_probabilities)
    auc_score = auc(fpr, tpr)

    # Plot and save ROC curve
    roc_fig = go.Figure([
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'), showlegend=False),
        go.Scatter(x=fpr, y=tpr, mode='lines', text=thresholds, showlegend=False,
                   hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{Logistic Regression Model - ROC Curve and AUC}}={auc_score:.3f}$",
                         width=400, height=400, xaxis=dict(title="FPR"), yaxis=dict(title="TPR")))
    roc_fig.write_image(f"./gd_lr_roc{lr:.4f}.png")

    # Select threshold that maximizes TPR-FPR and compute test error
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    logistic_model.alpha_ = optimal_threshold
    test_error = logistic_model.loss(X_test, y_test)

    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Test error with optimal threshold: {test_error}")

    # Perform cross-validation for L1-regularized logistic regression
    regularization_options = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    cross_val_scores = np.zeros((len(regularization_options), 2))
    for idx, lambda_val in enumerate(regularization_options):
        gd_solver = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
        reg_model = LogisticRegression(solver=gd_solver, penalty="l1", lam=lambda_val, alpha=0.5)
        cross_val_scores[idx] = cross_validate(estimator=reg_model, X=X_train, y=y_train,
                                               scoring=misclassification_error)


    # Plot cross-validation errors
    cv_fig = go.Figure([
        go.Scatter(x=regularization_options, y=cross_val_scores[:, 0], name="Cross-Validation Train Error"),
        go.Scatter(x=regularization_options, y=cross_val_scores[:, 1], name="Cross-Validation Validation Error")],
        layout=go.Layout(title="Cross-Validation Train and Validation errors",
                         xaxis=dict(title="Lambda", type="log"), yaxis_title="Error Value"))
    cv_fig.write_image(f"./l1_logistic_cross_validation_errors.png")

    # Fit the model with the best lambda
    best_lambda = regularization_options[np.argmin(cross_val_scores[:, 1])]
    gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
    final_model = LogisticRegression(solver=gd, penalty="l1", lam=best_lambda, alpha=0.5).fit(X_train, y_train)
    final_test_error = final_model.loss(X_test, y_test)

    print(f"Penalty: l1")
    print(f"Optimal lambda: {best_lambda}")
    print(f"Test error with optimal lambda: {final_test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
