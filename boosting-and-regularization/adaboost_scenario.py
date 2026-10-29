import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(wl=DecisionStump, iterations=n_learners)
    adaboost.fit(train_X, train_y)

    train_errors = []
    test_errors = []
    for T in range(1, n_learners + 1):
        train_errors.append(adaboost.partial_loss(train_X, train_y, T=T))
        test_errors.append(adaboost.partial_loss(test_X, test_y, T=T))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=train_errors, mode='lines', name='Training Error'))
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=test_errors, mode='lines', name='Test Error'))
    fig.update_layout(title=f'AdaBoost: Training and Test Errors (Noise = {noise})',
                      xaxis_title='Number of Learners',
                      yaxis_title='Misclassification Error')
    fig.write_image(f"Q1_train_test_errors_noise_{noise}.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=1, cols=4, subplot_titles=[rf"${x} $" for x in T])
    for idx, num_classifiers in enumerate(T):
        # Plot decision surface using decision_surface function
        decision_surface_trace = decision_surface(lambda X: adaboost.partial_predict(X, T=num_classifiers), lims[0],
                                                  lims[1], density=60, showscale=False)
        fig.add_trace(decision_surface_trace, row=1, col=idx + 1)

        # Plot test set
        test_trace = go.Scatter(
            x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
            marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x"), size=6)
        )
        fig.add_trace(test_trace, row=1, col=idx + 1)

    fig.update_layout(title_text=f'AdaBoost Decision Boundaries at Different Iteration Values (Noise = {noise})')
    fig.write_image(f"Q2_noise_{noise}_decision_boundaries.png")

    # Question 3: Decision surface of best performing ensemble
    best_T = np.argmin(test_errors) + 1  # +1 because T starts from 1
    predictions = adaboost.partial_predict(np.concatenate((train_X, test_X)), T=best_T)

    fig = make_subplots(rows=1, cols=1, subplot_titles=[
        f'AdaBoost Best Performing Ensemble (T={best_T}, Accuracy={1 - test_errors[best_T - 1]:.2f})'])

    # Plot decision surface of best performing ensemble
    decision_surface_trace = decision_surface(lambda X: adaboost.partial_predict(X, T=best_T), lims[0], lims[1],
                                              density=60, showscale=False)
    fig.add_trace(decision_surface_trace)

    # Plot test set
    test_trace = go.Scatter(
        x=test_X[:, 0], y=test_X[:, 1], mode='markers',
        marker=dict(color=test_y, colorscale='Viridis', size=4),
        showlegend=False
    )
    fig.add_trace(test_trace)
    fig.update_layout(
        title_text=f'AdaBoost Best Performing Ensemble (T={best_T}, Accuracy={1 - test_errors[best_T - 1]})')
    fig.write_image(f"Q3_best_ensemble_size_noise_{noise}.png")

    # Question 4: Decision surface with weighted samples
    weighted_samples = adaboost.D_ / np.max(adaboost.D_) * 5

    fig = make_subplots(rows=1, cols=1)

    # Plot decision surface of full ensemble
    decision_surface_trace = decision_surface(lambda X: adaboost.predict(X), lims[0], lims[1], density=60,
                                              showscale=False)
    fig.add_trace(decision_surface_trace)

    # Plot training set with weighted samples
    train_trace = go.Scatter(
        x=train_X[:, 0], y=train_X[:, 1], mode='markers',
        marker=dict(color=train_y, colorscale='Viridis', size=weighted_samples),
        showlegend=False
    )
    fig.add_trace(train_trace)
    fig.update_layout(title_text='AdaBoost Decision Surface with Weighted Samples')
    fig.write_image(f"Q4_last_iteration_training_set_noise_{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)

    # Execute for noise = 0
    noise = 0
    fit_and_evaluate_adaboost(noise)

    # Execute for noise = 0.4
    noise = 0.4
    fit_and_evaluate_adaboost(noise)