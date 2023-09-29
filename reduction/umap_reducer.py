import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from reduction.reducer import Reducer


class UMAPReducer(Reducer):
    def __init__(self, dimension: int = 2):
        super().__init__(dimension)

        self.pipe = make_pipeline(StandardScaler(), umap.UMAP(random_state=10, n_components=dimension))

    def reduce_dimension(self, features: np.array) -> np.array:
        return self.pipe.transform(features)

    def train(self, features: np.array):
        self.pipe.fit(features)
        self.trained = True
