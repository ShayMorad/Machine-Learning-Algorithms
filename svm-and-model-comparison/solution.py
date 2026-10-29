import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 69
SAMPLES_COUNT_1 = [5, 10, 20, 100]
SAMPLES_COUNT_2 = 200
C_REGULARIZATION = [0.1, 1, 5, 10, 100]
MOONS_NOISE = 0.2
CIRCLES_NOISE = 0.1
C1 = 1 / 5
C2 = 7
C3 = 5
MOONS = "moons"
CIRCLES = "circles"
GAUSSIANS = "gaussians"
TRAIN_TEST_SPLIT = 0.2


### HELPER FUNCTIONS ###
### PART 1 ###
def generate_samples_gauss_based(m: int, seed=SEED):
    # Define the mean and covariance matrix specified in the ex. description.
    mean = np.zeros(2)
    covariance_matrix = np.array([[1, 0.5], [0.5, 1]])

    # "Draw" the samples based on the given function by using NumPy's random generator
    random_generator: np.random.Generator = np.random.default_rng(seed)
    X_samples = random_generator.multivariate_normal(mean=mean, cov=covariance_matrix, size=m)
    W_weights = np.array([-0.6, 0.4])
    Y_labels = np.where(np.dot(X_samples, W_weights) >= 0, 1, -1)  # make sure there are only +1, -1 labels and not 0's.

    return X_samples, Y_labels


def svm_classification(X_samples: np.ndarray, Y_labels: np.ndarray, C: float) -> SVC:
    # Define the SVM learner with the parameters we want; C and for the linear problem.
    svm_classifier = SVC(C=C, kernel='linear')
    # Fit i.e. train on our data
    svm_classifier.fit(X_samples, Y_labels)
    return svm_classifier


def plot_1(m: int, C: float, svm_classifier: sklearn.svm.SVC, X_samples: np.ndarray, Y_labels: np.ndarray,
           save_path: str):
    # Create a canvas and plot
    canvas, axes = plt.subplots(figsize=(4, 4))
    # Split X to +1 and -1 groups by their labels
    X_true = X_samples[Y_labels == 1]
    X_neg = X_samples[Y_labels == -1]
    # Draw the samples with colors according to the true labels on the plot
    axes.scatter(*X_true.T, color="C2", label="+1")
    axes.scatter(*X_neg.T, color="C3", label="-1")
    # Draw the real decision boundary created by f and the decision boundary created by the SVM classifier
    W_real = np.array([-0.6, 0.4])
    min_x_scattered, max_x_scattered = axes.get_xlim()
    line = np.linspace(min_x_scattered, max_x_scattered)
    axes.plot(line, (-W_real[0] * line) / W_real[1], "y-", lw=2, label="True decision boundary by f")
    # Draw the decision boundary created by the SVM hypothesis
    B_learned = svm_classifier.intercept_[0]
    W_learned = svm_classifier.coef_[0]
    axes.plot(line, -(W_learned[0] * line - B_learned) / W_learned[1], "b--", lw=2,
              label="Predicted decision boundary by SVM")
    # Add information to the plot
    # axes.set_xlim(min_x_scattered, max_x_scattered)
    # axes.set_ylim(*axes.get_ylim())
    axes.legend(frameon=True, loc="center left", bbox_to_anchor=(1.03, 0.6), borderaxespad=0)

    axes.set_xlabel("x1")
    axes.set_ylabel("x2")
    axes.set_title(f"Amount of samples = {m}, Regularization parameter C = {C}")

    if save_path is None:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        canvas.savefig(os.path.join(save_path, f"plot_{m}samples_{C}regularization.png"), dpi=150, bbox_inches="tight")
        plt.close(canvas)
    return


### PART 2 ###
def generate_two_gaussians(m: int, seed=SEED):
    # Define the mean and covariance matrix specified in the ex. description.
    mean_positive = np.array([1, 1])
    mean_negative = np.array([-1, -1])
    covariance_matrix = np.array([[0.5, 0.2], [0.2, 0.5]])

    # "Draw" the samples based on the given function by using NumPy's random generator
    random_generator: np.random.Generator = np.random.default_rng(seed)
    X_samples_positive = random_generator.multivariate_normal(mean=mean_positive, cov=covariance_matrix, size=m // 2)
    X_samples_negative = random_generator.multivariate_normal(mean=mean_negative, cov=covariance_matrix,
                                                              size=m - (m // 2))
    X_samples = np.vstack([X_samples_positive, X_samples_negative])
    Y_labels = np.hstack([np.ones(len(X_samples_positive)), -np.ones(len(X_samples_negative))]).astype(np.int8)
    return X_samples, Y_labels


def get_all_datasets(m: int, seed=SEED):
    X_gaussians, Y_gaussians = generate_two_gaussians(m, SEED)
    X_moon, Y_moon = make_moons(n_samples=m, noise=MOONS_NOISE)
    Y_moon = np.where(Y_moon == 0, -1, 1).astype(np.int8)  # make sure labels that were 0 are -1 now
    X_circles, Y_circles = make_circles(n_samples=m, noise=CIRCLES_NOISE)
    Y_circles = np.where(Y_circles == 0, -1, 1).astype(np.int8)

    return X_gaussians, Y_gaussians, X_moon, Y_moon, X_circles, Y_circles


def make_classifiers(c1=C1, c2=C2, c3=C3):
    svm_classifier = SVC(kernel='linear', C=c1)
    decision_tree = DecisionTreeClassifier(max_depth=c2, random_state=SEED)
    knn = KNeighborsClassifier(n_neighbors=c3)
    return svm_classifier, decision_tree, knn


def plot_2(X_train, Y_train, X_test, Y_test, classifier, data_pool: str, save_path):
    padding = 0.4
    x_min, x_max = X_train[:, 0].min() - padding, X_train[:, 0].max() + padding
    y_min, y_max = X_train[:, 1].min() - padding, X_train[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # predict labels using classifier
    predictions = classifier[0].predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    canvas, axes = plt.subplots(figsize=(4.5, 4))
    # Background decision region
    axes.contourf(
        xx, yy, predictions,
        levels=[-1, 0, 1],
        colors=["#FFCCCC", "#CCE5FF"],  # light red, light blue
        alpha=0.35
    )

    # ----- scatter points -------------------------------------------
    axes.scatter(*X_train[Y_train == 1].T, c="C0", label="train +1", edgecolor="k", s=35)
    axes.scatter(*X_train[Y_train == -1].T, c="C1", label="train –1", edgecolor="k", s=35)
    axes.scatter(*X_test[Y_test == 1].T, c="C0", label="test +1", marker="x")
    axes.scatter(*X_test[Y_test == -1].T, c="C1", label="test –1", marker="x")

    acc = accuracy_score(Y_test, classifier[0].predict(X_test))
    axes.set_title(f"{data_pool} · {classifier[1]}\nacc = {acc:.2f}")
    axes.set_xlabel("x1")
    axes.set_ylabel("x2")
    axes.legend(frameon=True, loc="upper right")
    axes.set_aspect("equal", adjustable="box")

    # ----- save or show ---------------------------------------------
    if save_path is None:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        f_name = f"{data_pool}_{classifier[1]}.png"
        canvas.savefig(os.path.join(save_path, f_name), dpi=150, bbox_inches="tight")
        plt.close(canvas)


### Exercise Solution ###

def pratical_1_runner(save_path=None):
    for m in SAMPLES_COUNT_1:
        X_samples, Y_labels = generate_samples_gauss_based(m, SEED)
        while (len(np.unique(Y_labels)) != 2):
            X_samples, Y_labels = generate_samples_gauss_based(m, SEED)
        for C in C_REGULARIZATION:
            svm_classifier = svm_classification(X_samples, Y_labels, C)
            plot_1(m, C, svm_classifier, X_samples, Y_labels, save_path)


def practical_2_runner(save_path=None):
    X_gaussians, Y_gaussians, X_moon, Y_moon, X_circles, Y_circles = get_all_datasets(SAMPLES_COUNT_2, SEED)
    svm, decision, knn = make_classifiers(C1, C2, C3)
    X_gaussians_train, X_gaussians_test, Y_gaussians_train, Y_gaussians_test = train_test_split(X_gaussians,
                                                                                                Y_gaussians,
                                                                                                test_size=TRAIN_TEST_SPLIT,
                                                                                                stratify=Y_gaussians,
                                                                                                random_state=SEED)
    X_moon_train, X_moon_test, Y_moon_train, Y_moon_test = train_test_split(X_moon,
                                                                            Y_moon,
                                                                            test_size=TRAIN_TEST_SPLIT,
                                                                            stratify=Y_moon,
                                                                            random_state=SEED)
    X_circles_train, X_circles_test, Y_circles_train, Y_circles_test = train_test_split(X_circles,
                                                                                        Y_circles,
                                                                                        test_size=TRAIN_TEST_SPLIT,
                                                                                        stratify=Y_circles,
                                                                                        random_state=SEED)
    # copy classifiers for each data set and fit on them
    svm_2 = sklearn.base.clone(svm)
    svm_3 = sklearn.base.clone(svm)
    decision_2 = sklearn.base.clone(decision)
    decision_3 = sklearn.base.clone(decision)
    knn_2 = sklearn.base.clone(knn)
    knn_3 = sklearn.base.clone(knn)

    # fit all 3 classifiers [SVM, Decision, KNN] on Gaussians dataset
    svm.fit(X_gaussians_train, Y_gaussians_train)
    decision.fit(X_gaussians_train, Y_gaussians_train)
    knn.fit(X_gaussians_train, Y_gaussians_train)
    # fit all 3 classifiers [SVM, Decision, KNN] on Moons dataset
    svm_2.fit(X_moon_train, Y_moon_train)
    decision_2.fit(X_moon_train, Y_moon_train)
    knn_2.fit(X_moon_train, Y_moon_train)
    # fit all 3 classifiers [SVM, Decision, KNN] on Circles dataset
    svm_3.fit(X_circles_train, Y_circles_train)
    decision_3.fit(X_circles_train, Y_circles_train)
    knn_3.fit(X_circles_train, Y_circles_train)

    # plot all SVM plots for all 3 datasets
    plot_2(X_gaussians_train, Y_gaussians_train, X_gaussians_test, Y_gaussians_test, [svm, "SVM"], "Gaussians",
           save_path)
    plot_2(X_moon_train, Y_moon_train, X_moon_test, Y_moon_test, [svm_2, "SVM"], "Moons", save_path)
    plot_2(X_circles_train, Y_circles_train, X_circles_test, Y_circles_test, [svm_3, "SVM"], "Circles", save_path)

    # plot all Decision Tree plots for all 3 datasets
    plot_2(X_gaussians_train, Y_gaussians_train, X_gaussians_test, Y_gaussians_test, [decision, "Decision Tree"],
           "Gaussians",
           save_path)
    plot_2(X_moon_train, Y_moon_train, X_moon_test, Y_moon_test, [decision_2, "Decision Tree"], "Moons", save_path)
    plot_2(X_circles_train, Y_circles_train, X_circles_test, Y_circles_test, [decision_3, "Decision Tree"], "Circles",
           save_path)

    # plot all KNN plots for all 3 datasets
    plot_2(X_gaussians_train, Y_gaussians_train, X_gaussians_test, Y_gaussians_test, [knn, "KNN"], "Gaussians",
           save_path)
    plot_2(X_moon_train, Y_moon_train, X_moon_test, Y_moon_test, [knn_2, "KNN"], "Moons", save_path)
    plot_2(X_circles_train, Y_circles_train, X_circles_test, Y_circles_test, [knn_3, "KNN"], "Circles", save_path)


if __name__ == "__main__":
    np.random.seed(SEED)
    pratical_1_runner(save_path="plots/task1")
    practical_2_runner(save_path="plots/task2")
