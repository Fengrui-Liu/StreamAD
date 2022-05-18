from abc import ABC, abstractmethod

import numpy as np
from streamad.util import StreamStatistic


class BaseDetector(ABC):
    """Abstract class for Detector, supporting for customize detector."""

    def __init__(self):
        """Initialization BaseDetector"""
        self.data_type = "multivariate"
        self.index = -1
        self.window_len = 10
        self.score_stats = StreamStatistic()
        pass

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

    def fit_score(
        self,
        X: np.ndarray,
        normalized: bool = True,
        normalized_sigma: int = 3,
        normalized_global: bool = True,
    ) -> float:
        """Fit one observation and calculate its anomaly score.

        Args:
            X (np.ndarray): Data of current observation.
            normalized (bool, optional): Whether to normalize the score into a range of [0, 1]. Defaults to True.
            normalized_sigma (int, optional): We use k-sigma/z-score to report the anomalies, A large sigma inicates few anomalies. Defaults to 3.
            normalized_global (bool, optional): True for normalizing the score globally, with all history. Flase for normalizing the score within the window, with forgeting long histories. Defaults to True.

        Returns:
            float: Anomaly score. A high score indicates a high degree of anomaly.
        """

        if self.index == -1:
            if normalized_global:
                self.score_stats = StreamStatistic(is_global=True)
            else:
                self.score_stats = StreamStatistic(
                    is_global=False, window_len=self.window_len
                )
        self._check(X)

        score = self.fit(X).score(X)

        if score is None:
            return None

        score = float(score)

        if normalized:
            self.score_stats.update(score)
            score_mean = self.score_stats.get_mean()
            score_std = self.score_stats.get_std()
            sigma = np.divide(
                (score - score_mean),
                score_std,
                out=np.zeros_like(score),
                where=score_std != 0,
            )

            if sigma >= normalized_sigma:
                score_max = self.score_stats.get_max()
                score = (score - score_mean) / (score_max - score_mean)
            elif sigma <= -normalized_sigma:
                score_min = self.score_stats.get_min()
                score = (score - score_mean) / (score_min - score_mean)
            else:
                return 0

        return score
