from streamad.base import BaseDetector
import numpy as np
from collections import deque


class MadDetector(BaseDetector):
    def __init__(
        self, window_len: int = 10, threshold: float = 3.5, anomaly_window: int = 1
    ):
        """Median Absolute Deviation Detector :cite: `InfluxDB:MAD`.

        Args:
            window_len (int, optional): Length of sliding window. Defaults to 10.
            threshold (float, optional): threshold to decide a anomaly data. Defaults to 3.0.
            anomaly_window (int, optional): Length of sliding window of anomaly. only when all scores in the window are 1, then we consider this data as anomaly. Defaults to 1.

        parameters:
            scale_factor : Multiple relationship between standard deviation and absolute median difference under normal distribution.

        suggestions:
            this detector has high possibility of false positives. to reduce that you can set anomaly_window to 2 or 3 and set lower threshold like 3.0 or 2.5 to get better result.
        """
        super().__init__()
        self.window_len = window_len
        self.window = np.ndarray([0])
        self.threshold = threshold
        self.scale_factor = 1.4826
        self.ano_win = deque([0] * anomaly_window, maxlen=anomaly_window)

    def fit(self, X: np.ndarray) -> None:
        if self.window.size == self.window_len:
            self.window = np.delete(self.window, 0)
        self.window = np.append(self.window, X[0])
        self.cal_ano()
        return self

    def score(self, X: np.ndarray) -> float:
        if self.ano_win.count(0) == 0:
            return 1
        return 0

    def cal_ano(self):
        ori_median = np.median(self.window)
        abs_diff = np.abs(self.window - ori_median)
        mad = self.scale_factor * np.median(abs_diff)
        self.ano_win.pop()
        self.ano_win.insert(0, 1 if abs_diff[-1] / mad > self.threshold else 0)
        return
