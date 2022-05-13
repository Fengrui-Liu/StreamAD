import random
from typing import Union

import numpy as np
import pandas as pd
from streamad.base import BaseDetector


class RandomDetector(BaseDetector):
    """Return random anomaly score. A minimum score for benchmark."""

    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray) -> None:
        """Detector fit current observation from StreamGenerator.

        Args:
            X (np.ndarray): The data value of current observation from StreamGenerator.
        """
        return self

    def score(self, X: np.ndarray) -> float:
        """Abstract method: Detector predict the probability of anomaly for current observation form StreamGenerator.

        Args:
            X (np.ndarray): The data value of current observation from StreamGenerator.

        Returns:
            float: Anomaly probability.
        """

        return random.random()
