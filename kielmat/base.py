from abc import ABC, abstractmethod
import numpy as np


class BaseAlgorithm(ABC):

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        pass