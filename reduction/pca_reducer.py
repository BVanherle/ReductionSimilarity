import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from reduction.reducer import Reducer


class PCAReducer(Reducer):
    def __init__(self, dimension: int = 2):
        super().__init__(dimension)
        self.pipe = make_pipeline(StandardScaler(), PCA(n_components=dimension))

    def train(self, features: np.array):
        self.pipe.fit(features)

    def reduce_dimension(self, features: np.array) -> np.array:
        self.pipe.transform(features)
