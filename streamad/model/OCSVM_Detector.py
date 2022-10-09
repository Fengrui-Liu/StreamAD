from sklearn.svm import OneClassSVM
import numpy as np
from streamad.base.detector import BaseDetector
from collections import deque
from typing import Literal


class OCSVMDetector(BaseDetector):
    def __init__(
        self,
        nu: float = 0.5,
        kernel: Literal[
            "linear", "poly", "rbf", "sigmoid", "precomputed"
        ] = "rbf",
        **kwargs
    ):
        """One-Class SVM Detector :cite:`enwiki:1098733917`.

        Args:
            nu (float, optional): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Defaults to 0.5.
            kernel (str, optional): The kernel type to be used in the algorithm. Defaults to "rbf".
        """
        super().__init__(data_type="multivariate", **kwargs)
        self.nu = nu
        self.kernel = kernel
        self.model = None

    def fit(self, X: np.ndarray, timestamp: int = None):

        self.window.append(X)
        if self.index >= self.window_len:
            self.model = OneClassSVM(
                gamma="scale", nu=self.nu, kernel=self.kernel
            )
            self.model.fit(list(self.window))

        return self

    def score(self, X: np.ndarray, timestamp: int = None):

        score = self.model.decision_function(X.reshape(1, -1))
        return abs(score)
