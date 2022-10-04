from abc import ABC, abstractmethod
import numpy as np


class Reducer(ABC):
    def __init__(self):
        self.trained = False

    @abstractmethod
    def train(self, features: np.array):
        pass

    @abstractmethod
    def reduce_dimension(self, features: np.array) -> np.array:
        pass

