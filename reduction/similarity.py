import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline


def kernel_density_overlap(set_a: np.array, set_b: np.array) -> float:
    kde = KernelDensity(kernel='gaussian').fit(set_a)
    scores = kde.score_samples(set_b)
    norm = np.linalg.norm(-scores)

    return np.mean(-scores/norm)


def gaussian_mixture_overlap(set_a: np.array, set_b: np.array) -> float:
    gmm = GaussianMixture().fit(set_a)

    scores = gmm.score_samples(set_b)
    norm = np.linalg.norm(-scores)

    return np.mean(-scores / norm)


def svm_overlap(set_a: np.array, set_b: np.array) -> float:
    transform = Nystroem()

    svm = linear_model.SGDOneClassSVM(shuffle=True, fit_intercept=True)
    svm_pipe = make_pipeline(transform, svm)
    svm_pipe.fit(set_a)
    scores = svm_pipe.score_samples(set_b)

    return np.mean(scores)


if __name__ == "__main__":
    set_a = [[0.1, 1.0], [0.3, 1.0], [0.4, 1.0], [1.0, 1.0]]
    set_b = [[0.7, 1.0], [1.2, 1.0], [1.4, 1.0]]

    print(svm_overlap(set_a, set_b))