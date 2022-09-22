from streamad.base import BaseDetector
import numpy as np
from math import log
from scipy.optimize import minimize
from collections import deque
import heapq


class ZSpotDetector(BaseDetector):
    def __init__(
        self,
        back_mean_len: int = 20,
        num_over_threshold: int = 10,
        deviance_ratio: float = 0.01,
        z: int = 2,
        **kwargs
    ):

        super().__init__(data_type="univariate", **kwargs)

        self.deviance_ratio = deviance_ratio

        self.back_mean_len = back_mean_len

        self.back_mean_window = deque(maxlen=max(self.back_mean_len, 2))

        self.num_over_threshold = num_over_threshold

        nonedict = {"up": None, "down": None}

        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.history_peaks = {"up": [], "down": []}
        self.normal_X = None
        self.z = z

    def _update_oneside(self, side: str):

        if side == "up":
            self.init_threshold[side] = heapq.heappushpop(
                self.history_peaks[side], self.normal_X
            )
            self.peaks[side] = np.array(self.history_peaks[side]) - np.array(
                self.init_threshold[side]
            )
            std = np.sqrt(
                np.sum([i**2 for i in self.peaks[side]])
                / self.num_over_threshold
            )
            self.extreme_quantile[side] = (
                self.init_threshold[side] + self.z * std
            )
        elif side == "down":
            self.init_threshold[side] = - heapq.heappushpop(
                self.history_peaks[side], -self.normal_X
            )
            self.peaks[side] = np.array(self.history_peaks[side]) + np.array(
                self.init_threshold[side]
            )
            std = np.sqrt(
                np.sum([i**2 for i in self.peaks[side]])
                / self.num_over_threshold
            )
            self.extreme_quantile[side] = (
                self.init_threshold[side] - self.z * std
            )

        else:
            raise NotImplementedError

    def _cal_back_mean(self, X):

        back_mean = np.array(0)

        if self.back_mean_len == 1:
            # least back_mean_window is 2
            back_mean = self.back_mean_window[-1]
        elif self.back_mean_len > 1:
            back_mean = np.mean(self.back_mean_window)

        return X - back_mean

    def fit(self, X: np.ndarray):
        """Fit the data to the detector.

        Args:
            X (np.ndarray): Data of current observation.
        """
        X = float(X[0])

        if self.index >= self.back_mean_len:
            self.normal_X = self._cal_back_mean(X)

        if (
            self.back_mean_len
            <= self.index
            < self.num_over_threshold + self.back_mean_len
        ):
            heapq.heappush(self.history_peaks["up"], self.normal_X)
            # We use negative x to simulate a maxheap
            heapq.heappush(self.history_peaks["down"], -self.normal_X)

        elif self.index == self.num_over_threshold + self.back_mean_len:

            self._update_oneside("up")
            self._update_oneside("down")

        elif self.index > self.num_over_threshold + self.back_mean_len:
            if self.normal_X > self.init_threshold["up"]:
                self._update_oneside("up")

            elif self.normal_X < self.init_threshold["down"]:
                self._update_oneside("down")

        self.back_mean_window.append(X)
        return self

    def score(self, X: np.ndarray) -> float:

        curr_X = self.back_mean_window[-1]
        last_X = self.back_mean_window[-2]

        if (
            abs(
                np.divide(
                    curr_X - last_X, last_X, np.array(curr_X), where=last_X != 0
                )
            )
            < self.deviance_ratio
        ):
            score = 0.0

        elif (
            self.normal_X > self.extreme_quantile["up"]
            or self.normal_X < self.extreme_quantile["down"]
        ):
            score = 1.0

        elif self.normal_X > self.init_threshold["up"]:
            side = "up"
            score = np.divide(
                self.normal_X - self.init_threshold[side],
                (self.extreme_quantile[side] - self.init_threshold[side]),
                np.array(0.9),
                where=(
                    self.extreme_quantile[side] - self.init_threshold[side] != 0
                ),
            )
        elif self.normal_X < self.init_threshold["down"]:
            side = "down"
            score = np.divide(
                self.init_threshold[side] - self.normal_X,
                (self.init_threshold[side] - self.extreme_quantile[side]),
                np.array(0.5),
                where=(
                    self.init_threshold[side] - self.extreme_quantile[side] != 0
                ),
            )
        else:
            score = 0.0

        return float(score)
