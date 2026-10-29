from classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import matplotlib.pyplot as plt
from math import atan2, pi
from loss_functions import accuracy

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        Perceptron(callback=lambda percept, _, __: losses.append(percept.loss(X, y))).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plt.figure()
        plt.plot(range(len(losses)), losses, color='black', linestyle='-')
        plt.title(f'Perceptron Training Loss - {n}')
        plt.xlabel('Fitting Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{n.replace(' ', '_')}_loss_graph.png")
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        X, y = load_dataset(f"{f}")

        gnb = GaussianNaiveBayes()
        lda = LDA()

        gnb.fit(X, y)
        lda.fit(X, y)

        gnb_prediction = gnb.predict(X)
        lda_prediction = lda.predict(X)

        gnb_acc = accuracy(y, gnb_prediction)
        lda_acc = accuracy(y, lda_prediction)

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Get unique class labels
        unique_classes = np.unique(y)

        # Define markers dynamically based on the number of unique classes
        markers = ['o', 'x', 'D', '^', 's', '*', 'P', 'h', '+', 'v', '<', '>'][:len(unique_classes)]

        # Define a color map with distinct colors for each class
        color_map = plt.cm.get_cmap('viridis', len(unique_classes))
        colors = [color_map(i) for i in range(len(unique_classes))]

        # Plot Gaussian Naive Bayes predictions
        for i, true_class in enumerate(unique_classes):
            mask = (y == true_class)
            axes[0].scatter(X[mask, 0], X[mask, 1], c=[colors[cls] for cls in gnb_prediction[mask]], marker=markers[i],
                            label=f'Class {true_class} (Predicted)')

            # Plot Gaussian center
            axes[0].scatter(gnb.mu_[true_class, 0], gnb.mu_[true_class, 1], marker='X', color='black', s=100,
                            label='Gaussian Center')

            # Plot ellipse for Gaussian
            ellipse_trace = get_ellipse(gnb.mu_[true_class], np.diag(gnb.vars_[true_class]))
            axes[0].plot(ellipse_trace.x, ellipse_trace.y, color='black')

        axes[0].set_title(f"Gaussian Naive Bayes (accuracy = {np.round(100 * gnb_acc, 2)}%)")
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].grid(True)

        # Plot LDA predictions
        for i, true_class in enumerate(unique_classes):
            mask = (y == true_class)
            axes[1].scatter(X[mask, 0], X[mask, 1], c=[colors[cls] for cls in lda_prediction[mask]], marker=markers[i],
                            label=f'Class {true_class} (Predicted)')

            # Plot Gaussian center
            axes[1].scatter(lda.mu_[true_class, 0], lda.mu_[true_class, 1], marker='X', color='black', s=100,
                            label='Gaussian Center')

            # Plot ellipse for Gaussian
            ellipse_trace = get_ellipse(lda.mu_[true_class], lda.cov_)
            axes[1].plot(ellipse_trace.x, ellipse_trace.y, color='black')

        axes[1].set_title(f"LDA (accuracy = {np.round(100 * lda_acc, 2)}%)")
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].grid(True)

        fig.suptitle(f'Dataset: {f}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"gnb_vs_lda_{f.replace(' ', '_').replace('.npy', '')}.png")
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
