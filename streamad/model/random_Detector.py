import random

import numpy as np
from streamad.base import BaseDetector


class RandomDetector(BaseDetector):
    """Return random anomaly score. A minimum score for benchmark."""

    def __init__(self, **kwargs):
        super().__init__(data_type="multivariate", **kwargs)

    def fit(self, X: np.ndarray) -> None:
        return self

    def score(self, X: np.ndarray) -> float:

        return random.random()
