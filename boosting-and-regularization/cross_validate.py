from typing import Tuple
import numpy as np
from base_estimator import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data. Has functions: fit, predict, loss

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples = X.shape[0]
    fold_sizes = np.full(cv, n_samples // cv)
    fold_sizes[:n_samples % cv] += 1  # Distribute remainder, If n_samples = 103 and cv = 5,
                                      # we'll get folds of sizes [21, 21, 21, 20, 20].

    # Step 2: split into k equally sized sets S1,...,Sk
    folds = []
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        folds.append(np.arange(start, end))
        start = end

    train_losses = []
    validation_losses = []

    # Step 3a: for i = 1,...,k
    for i in range(cv):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(cv) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # hi = A(S \ S_i)
        model = estimator
        model.fit(X_train, y_train)

        # Errors on both train and validation
        train_loss = model.loss(X_train, y_train)
        val_loss = model.loss(X_val, y_val)

        train_losses.append(train_loss)
        validation_losses.append(val_loss)

    # Step 3b: Return average losses
    return float(np.mean(train_losses)), float(np.mean(validation_losses))
