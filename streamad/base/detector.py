from abc import ABC, abstractmethod

import numpy as np
from streamad.util import StreamStatistic


class BaseDetector(ABC):
    """Abstract class for Detector, supporting for customize detector."""

    def __init__(self):
        """Initialization BaseDetector"""
        self.data_type = "multivariate"
        self.index = -1
        self.window_len = 100

    def _check(self, X) -> bool:
        """Check whether the detector can handle the data."""
        x_shape = X.shape[0]

        if self.data_type == "univariate":
            assert x_shape == 1, "The data is not univariate."
        elif self.data_type == "multivariate":
            assert x_shape >= 1, "The data is not univariate or multivariate."

        self.index += 1

    @abstractmethod
    def fit(self, X: np.ndarray):

        return NotImplementedError

    @abstractmethod
    def score(self, X: np.ndarray) -> float:

        return NotImplementedError

    def fit_score(self, X: np.ndarray) -> float:
        """Fit one observation and calculate its anomaly score.

        Args:
            X (np.ndarray): Data of current observation.
            normalized (bool, optional): Whether to normalize the score into a range of [0, 1]. Defaults to True.
            normalized_sigma (int, optional): We use k-sigma/z-score to report the anomalies, A large sigma inicates few anomalies. Defaults to 3.
            normalized_global (bool, optional): True for normalizing the score globally, with all history. Flase for normalizing the score within the window, with forgeting long histories. Defaults to True.

        Returns:
            float: Anomaly score. A high score indicates a high degree of anomaly.
        """

        self._check(X)
        if self.index < self.window_len:
            self.fit(X)
            return None

        score = self.fit(X).score(X)

        return float(score)
