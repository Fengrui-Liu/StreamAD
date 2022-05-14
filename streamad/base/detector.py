from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Abstract class for Detector, supporting for customize detector."""

    def __init__(self):
        """Initialization BaseDetector"""
        self.data_type = "multivariate"
        pass

    def _check(self, X) -> bool:
        """Check whether the detector can handle the data."""
        x_shape = X.shape[0]

        if self.data_type == "univariate":
            assert x_shape == 1, "The data is not univariate."
        elif self.data_type == "multivariate":
            assert x_shape >= 1, "The data is not univariate or multivariate."

    @abstractmethod
    def fit(self, X: np.ndarray,) -> None:

        return NotImplementedError

    @abstractmethod
    def score(self, X: np.ndarray) -> float:

        return NotImplementedError

    def fit_score(self, X: np.ndarray) -> float:
        """Detector fit and score the anomaly of current observation from StreamGenerator.

        Args:
            X (np.ndarray): Data of current observation

        Returns:
            float: Anomaly score. 1.0 for anomaly and 0.0 for normal.
        """
        self._check(X)

        return self.fit(X).score(X)
