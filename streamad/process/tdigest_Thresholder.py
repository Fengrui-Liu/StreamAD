from streamad.base import BaseDetector
import numpy as np
from typing import Type
from tdigest import TDigest
from collections import deque


class TDigestThresholder:
    def __init__(
        self,
        detector: BaseDetector,
        percentile_up: float = 95,
        percentile_down: float = 5,
        is_global: bool = True,
        window_len: int = 100,
    ) -> None:
        """A thresholder which can filter out outliers using t-digest, and normalize the anomaly scores into [0,1] :cite:`DBLP:journals/simpa/Dunning21`.

        Args:
            detector (BaseDetector): A detector that must be a child class of BaseDetector.
            percentile_up (float, optional): We regard the scores above `percentile_up` as anomalies. Defaults to 95.
            percentile_down (float, optional): We regard the scores below `percentile_down` as anomalies. Defaults to 5.
            is_global (bool, optional): Method to record, a global way or a rolling window way. Defaults to True.
            window_len (int, optional): The length of rolling window, ignore this when `is_global=True`. Defaults to 100.
        """
        self.detector = detector
        self.percentile_up = percentile_up
        self.percentile_down = percentile_down
        self.init_data = []
        self.init_flag = False

        assert (
            percentile_up >= 0
            and percentile_up <= 100
            and percentile_down >= 0
            and percentile_down <= 100
        ), "percentile must be between 0 and 100"

        self.is_global = is_global
        self.score_stats = TDigest()
        self.score_deque = (
            deque(maxlen=detector.window_len)
            if is_global
            else deque(maxlen=window_len)
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
                self.score_deque.append(score)

            self.score_stats.batch_update(self.score_deque)

        score = self.detector.fit_score(X)

        if self.is_global:
            self.score_stats.update(score)
        else:
            self.score_stats = TDigest()
            self.score_stats.batch_update(self.score_deque)

        percentile_up = self.score_stats.percentile(self.percentile_up)
        percentile_down = self.score_stats.percentile(self.percentile_down)

        if score > percentile_up or score < percentile_down:
            score = 1.0
        else:
            score = 0.0

        return score
