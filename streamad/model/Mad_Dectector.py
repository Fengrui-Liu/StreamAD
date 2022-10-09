from streamad.base import BaseDetector
import numpy as np
from collections import deque


class MadDetector(BaseDetector):
    def __init__(self, **kwargs):
        """Median Absolute Deviation Detector :cite: `InfluxDB:MAD`.

        Args:
            window_len (int, optional): Length of sliding window. Defaults to 10.
            threshold (float, optional): threshold to decide a anomaly data. Defaults to 3.0.

        parameters:
            scale_factor : Multiple relationship between standard deviation and absolute median difference under normal distribution.

        """
        super().__init__(data_type="univariate", **kwargs)
        self.scale_factor = 1.4826

    def fit(self, X: np.ndarray, timestamp: int = None):
        self.window.append(X[0])

        return self

    def score(self, X: np.ndarray, timestamp: int = None):

        ori_median = np.median(self.window)
        abs_diff = np.abs(self.window - ori_median)
        mad = self.scale_factor * np.median(abs_diff)
        score = np.divide(
            abs_diff[-1], mad, out=np.array(abs_diff[-1] / 1e-5), where=mad != 0
        )

        return score
