from typing import Type

import numpy as np
from streamad.base import BaseDetector
from streamad.util import StreamStatistic


class ZScoreThresholder:
    def __init__(
        self,
        detector: BaseDetector,
        sigma: int = 3,
        is_global: bool = True,
        window_len: int = 100,
    ) -> None:
        """A thresholder which can filter out outliers using z-score, and normalize the anomaly scores into [0,1].

        Args:
            detector (BaseDetector): A detector that must be a child class of BaseDetector.
            sigma (int, optional): Zscore threshold, we regard the scores out of sigma as anomalies. Defaults to 3.
            is_global (bool, optional): Method to record, a global way or a rolling window way. Defaults to True.
            window_len (int, optional): The length of rolling window, ignore this when `is_global=True`. Defaults to 100.
        """
        self.detector = detector
        self.sigma = sigma
        self.init_data = []
        self.init_flag = False
        self.score_stats = StreamStatistic(
            is_global=is_global, window_len=window_len
        )

    def fit_score(self, X: np.ndarray) -> float:
        if self.detector.index < self.detector.window_len:
            self.init_data.append(X)
            self.detector.fit_score(X)
            return None

        if not self.init_flag:
            self.init_flag = True
            for data in self.init_data:
                score = self.detector.score(data)
                self.score_stats.update(score)

        score = self.detector.fit_score(X)

        self.score_stats.update(score)

        score_mean = self.score_stats.get_mean()
        score_std = self.score_stats.get_std()

        sigma = np.divide(
            (score - score_mean),
            score_std,
            out=np.zeros_like(score),
            where=score_std != 0,
        )

        if sigma >= self.sigma:
            score_max = self.score_stats.get_max()
            score = (score - score_mean) / (score_max - score_mean)
        elif sigma <= -self.sigma:
            score_min = self.score_stats.get_min()
            score = (score - score_mean) / (score_min - score_mean)
        else:
            return 0

        return score
