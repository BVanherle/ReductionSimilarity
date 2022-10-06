from abc import ABC, abstractmethod
import numpy as np


class Reducer(ABC):
    def __init__(self, dimension):
        self.trained = False
        self.dimension = dimension

    @abstractmethod
    def train(self, features: np.array):
        pass

    @abstractmethod
    def reduce_dimension(self, features: np.array) -> np.array:
        pass

