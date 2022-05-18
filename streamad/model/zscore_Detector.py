from streamad.base import BaseDetector
import numpy as np


class ZScoreDetector(BaseDetector):
    def __init__(self):
        super().__init__()

        self.data_type = "univariate"

    def fit(self, X: np.ndarray):
        return self

    def score(self, X: np.ndarray):

        return X[0]
