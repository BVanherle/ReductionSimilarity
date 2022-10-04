import numpy as np
import umap

from reduction.reducer import Reducer


class UMAPReducer(Reducer):
    def __init__(self):
        super().__init__()

        self.umap_reducer = umap.UMAP(random_state=10)

    def reduce_dimension(self, features: np.array) -> np.array:
        return self.umap_reducer.transform(features)

    def train(self, features: np.array):
        self.umap_reducer.fit(features)
        self.trained = True
