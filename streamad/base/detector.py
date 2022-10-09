from abc import ABC, abstractmethod

import numpy as np
from collections import deque


class BaseDetector(ABC):
    """Abstract class for Detector, supporting for customize detector."""

    def __init__(
        self,
        window_len: int = 100,
        detrend: bool = False,
        detrend_len: int = 20,
        data_type: str = "multivariate",
    ):
        """Initialize the attributes of the BaseDetector class


        Args:
            window_len (int, optional): Length of window for observations. Defaults to 100.
            detrend (bool, optional): Data is detrended by subtracting the mean. Defaults to True.
            detrend_len (int, optional): Length of data for reference to detrend. Defaults to 20.
            data_type (str, optional): Multi/Univariate data type. Defaults to "multivariate".
        """

        self.data_type = data_type
        self.index = -1
        self.detrend = detrend
        self.window_len = window_len
        self.detrend_len = detrend_len
        self.window = deque(maxlen=self.window_len)
        self.detrend_window = deque(maxlen=self.detrend_len)

    def _check(self, X) -> bool:
        """Check whether the detector can handle the data."""
        x_shape = X.shape[0]

        if self.data_type == "univariate":
            assert x_shape == 1, "The data is not univariate."
        elif self.data_type == "multivariate":
            assert x_shape >= 1, "The data is not univariate or multivariate."

        if np.isnan(X).any():
            return False
        self.index += 1
        return True

    def _detrend(self, X: np.ndarray) -> np.ndarray:
        """Detrend the data by subtracting the mean.

        Args:
            X (np.ndarray): Data of current observation.

        Returns:
            np.ndarray: Detrended data.
        """

        self.detrend_window.append(X)

        return X - np.mean(self.detrend_window, axis=0)

    @abstractmethod
    def fit(self, X: np.ndarray, timestamp=None):

        return NotImplementedError

    @abstractmethod
    def score(self, X: np.ndarray, timestamp=None) -> float:

        return NotImplementedError

    def fit_score(self, X: np.ndarray, timestamp=None) -> float:
        """Fit one observation and calculate its anomaly score.

        Args:
            X (np.ndarray): Data of current observation.

        Returns:
            float: Anomaly score. A high score indicates a high degree of anomaly.
        """

        check_flag = self._check(X)
        if not check_flag:
            return None
        X = self._detrend(X) if self.detrend else X

        if self.index < self.window_len:
            self.fit(X,timestamp)
            return None

        score = self.fit(X, timestamp).score(X, timestamp)

        return float(abs(score))
