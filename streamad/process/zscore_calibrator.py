import numpy as np
from streamad.util import StreamStatistic


class ZScoreCalibrator:
    def __init__(
        self,
        sigma: int = 3,
        extreme_sigma: int = 5,
        is_global: bool = True,
        window_len: int = 100,
    ) -> None:
        """A calibrator which can filter out outliers using z-score, and normalize the anomaly scores into [0,1].

        Args:
            sigma (int, optional): Zscore threshold, we regard the scores out of sigma as potential anomalies. Defaults to 2.
            extreme_sigma (int, optional): Zscore threshold for extreme values, we regard the scores out of extreme_sigma as extreme anomalies. Defaults to 3.
            is_global (bool, optional): Method to record, a global way or a rolling window way. Defaults to True.
            window_len (int, optional): The length of rolling window, ignore this when `is_global=True`. Defaults to 100.
        """
        self.sigma = sigma
        self.extreme_sigma = extreme_sigma
        self.init_data = []
        self.init_flag = False
        self.score_stats = StreamStatistic(
            is_global=is_global, window_len=window_len
        )

    def normalize(self, score: float) -> float:
        if score is None:
            return None

        self.score_stats.update(score)

        score_mean = self.score_stats.get_mean()
        score_std = self.score_stats.get_std()

        sigma = np.divide(
            (score - score_mean),
            score_std,
            out=np.array((score - score_mean) / 1e-5),
            where=score_std != 0,
        )
        sigma = abs(sigma)

        if sigma > self.extreme_sigma:
            return 1.0
        elif sigma > self.sigma:
            score_max = self.score_stats.get_max()
            score = np.divide(
                (score - score_mean),
                (score_max - score_mean),
                out=max(np.array((score - score_mean) / 1e-5), np.array(1.0)),
                where=score_max != score_mean,
            )
            score = abs(score)
        else:
            return 0.0

        return score
