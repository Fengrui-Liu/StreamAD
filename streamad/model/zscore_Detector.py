from streamad.base import BaseDetector
import numpy as np
from streamad.util import StreamStatistic


class ZScoreDetector(BaseDetector):
    def __init__(self, is_global: bool = False, **kwargs):
        """Univariate Z-Score Detecto :cite:`enwiki:1086685336`

        Args:
            window_len (int, optional):  Length of the window for reference. Defaults to 50.
            is_global (bool, optional): Whether to detect anomalies from a global view. Defaults to False.
        """
        super().__init__(data_type="univariate", **kwargs)

        self.stat = StreamStatistic(
            is_global=is_global, window_len=self.window_len
        )

    def fit(self, X: np.ndarray, timestamp: int = None):
        self.stat.update(X[0])
        return self

    def score(self, X: np.ndarray, timestamp: int = None):
        mean = self.stat.get_mean()
        std = self.stat.get_std()

        score = np.divide(
            (X[0] - mean), std, out=np.zeros_like(X[0]), where=std != 0
        )

        return score
