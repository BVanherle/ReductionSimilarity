import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
import reduction.prd_score


def pdr_overlap(set_a: np.array, set_b: np.array) -> float:
    p, r = reduction.prd_score.compute_prd_from_embedding(set_b, set_a)
    return np.mean(r)


def kernel_density_overlap(set_a: np.array, set_b: np.array) -> float:
    kde = KernelDensity(kernel='gaussian').fit(set_a)
    scores = kde.score_samples(set_b)

    return np.mean(scores)


def gaussian_mixture_overlap(set_a: np.array, set_b: np.array) -> float:
    gmm = GaussianMixture().fit(set_a)

    scores = gmm.score_samples(set_b)

    return np.mean(scores)


def svm_overlap(set_a: np.array, set_b: np.array) -> float:
    transform = Nystroem()

    svm = linear_model.SGDOneClassSVM()
    svm_pipe = make_pipeline(transform, svm)
    svm_pipe.fit(set_a)
    scores = svm_pipe.score_samples(set_b)

    return np.mean(scores)


if __name__ == "__main__":
    set_a = [[0.1, 1.0], [0.3, 1.0], [0.4, 1.0], [1.0, 1.0]]
    set_b = [[0.7, 1.0], [1.2, 1.0], [1.4, 1.0]]

    print(svm_overlap(set_a, set_b))
