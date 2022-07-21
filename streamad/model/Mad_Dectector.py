from streamad.base import BaseDetector
import numpy as np


class MadDetector(BaseDetector):
    def __init__(self, window_len: int = 10, threshold: float = 3.0):
        """Median Absolute Deviation Detector :reference: https://www.influxdata.com/blog/anomaly-detection-with-median-absolute-deviation/#:~:text=How%20Median%20Absolute%20Deviation%20algorithm,time%20series%20at%20that%20timestamp

        Args:
            window_len (int, optional): Length of sliding window. Defaults to 10.
            threshold (float, optional): threshold to decide a anomaly data. Defaults to 3.0.

        parameters:
            scale_factor : Multiple relationship between standard deviation and absolute median difference under normal distribution.
        """
        super().__init__()
        self.window_len = window_len
        self.window = np.ndarray
        self.threshold = threshold
        self.scale_factor = 1.4826

    def fit(self, X: np.ndarray) -> None:
        if self.window.size == self.window_len:
            self.window = np.delete(self.window, 0)
        self.window = np.append(self.window, X[0])
        return self

    def score(self, X: np.ndarray) -> float:
        return self.cal_ano()

    def cal_ano(self):
        ori_median = np.median(self.window)
        abs_diff = np.abs(self.window - ori_median)
        mad = self.scale_factor * np.median(abs_diff)
        return 1 if abs_diff[-1] / mad > self.threshold else 0
