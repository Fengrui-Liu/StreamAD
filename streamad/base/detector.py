from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class BaseDetector(ABC):
    """Abstract class for Detector, supporting for customize detector."""

    def __init__(self):
        """Initialization BaseDetector"""
        pass

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> None:
        """Detector fit current observation from StreamGenerator.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The data value of current observation from StreamGenerator.
        """
        return self

    @abstractmethod
    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """Detector score the probability of anomaly for current observation form StreamGenerator.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The data value of current observation from StreamGenerator.

        Returns:
            float: Anomaly probability. 1.0 for anomaly and 0.0 for normal.
        """

        return 0.0

    def fit_score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """Detector fit and score the probability of anomaly current observation from StreamGenerator.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): [description]

        Returns:
            float: [description]
        """

        return self.fit(X).score(X)