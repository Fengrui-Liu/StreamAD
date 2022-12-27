import datetime
import heapq
from collections import deque
from copy import deepcopy

import numpy as np
from streamad.base import BaseDetector


class ZSpotDetector(BaseDetector):
    def __init__(
        self,
        back_mean_len: int = 20,
        num_over_threshold: int = 30,
        deviance_ratio: float = 0.01,
        z: int = 2,
        expire_days: int = 14,
        ignore_n: int = 10,
        **kwargs
    ):

        super().__init__(data_type="univariate", **kwargs)

        self.deviance_ratio = deviance_ratio

        self.back_mean_len = back_mean_len
        self.back_mean_window = deque(maxlen=max(self.back_mean_len, 2))

        self.num_over_threshold = num_over_threshold

        nonedict = {"up": None, "down": None}

        self.extreme_quantile = dict.copy(nonedict)
        self.local_init_threshold = dict.copy(nonedict)
        self.global_init_threshold = dict.copy(nonedict)

        self.last_date = None

        self.date = deque(maxlen=expire_days)
        self.date_peaks = deque(maxlen=expire_days)

        self.history_peaks = {"up": [], "down": []}
        self.normal_X = None
        self.time_X = None
        self.z = z
        self.ignore_n = ignore_n

    def _update_oneside(self, side: str, init: bool = False):
        if side == "up":
            if init is False:
                self.local_init_threshold[side] = heapq.heappushpop(
                    self.history_peaks[side], self.normal_X
                )
            else:
                self.local_init_threshold[side] = self.history_peaks[side][0]

            peaks = deepcopy(self.history_peaks[side])
            for i in self.date_peaks:
                peaks.extend(i[side])

            selected_peaks = heapq.nlargest(self.num_over_threshold, peaks)
            self.global_init_threshold[side] = selected_peaks[-1]
            selected_peaks = np.array(selected_peaks) - np.array(
                self.global_init_threshold[side]
            )
            std = np.sqrt(
                np.sum([i**2 for i in selected_peaks])
                / self.num_over_threshold
            )
            self.extreme_quantile[side] = (
                self.global_init_threshold[side] + self.z * std
            )
        elif side == "down":

            if init is False:
                self.local_init_threshold[side] = -heapq.heappushpop(
                    self.history_peaks[side], -self.normal_X
                )
            else:
                self.local_init_threshold[side] = -self.history_peaks[side][0]

            peaks = deepcopy(self.history_peaks[side])
            for i in self.date_peaks:
                peaks.extend(i[side])

            selected_peaks = heapq.nlargest(self.num_over_threshold, peaks)
            self.global_init_threshold[side] = -selected_peaks[-1]
            selected_peaks = np.array(selected_peaks) + np.array(
                self.global_init_threshold[side]
            )
            std = np.sqrt(
                np.sum([i**2 for i in selected_peaks])
                / self.num_over_threshold
            )
            self.extreme_quantile[side] = (
                self.global_init_threshold[side] - self.z * std
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

    def fit(self, X: np.ndarray, timestamp=None):
        """Fit the data to the detector.

        Args:
            X (np.ndarray): Data of current observation.
        """
        X = float(X[0])

        if self.index >= self.back_mean_len + self.ignore_n:
            self.normal_X = self._cal_back_mean(X)
            self.time_X = datetime.datetime.fromtimestamp(timestamp)

            if self.last_date is None:
                self.last_date = self.time_X.date()
                self.history_peaks["up"] = [self.normal_X]
                self.history_peaks["down"] = [-self.normal_X]

            elif self.last_date != self.time_X.date():
                self.date.append(self.last_date)
                self.date_peaks.append(deepcopy(self.history_peaks))
                self.last_date = self.time_X.date()
                self.history_peaks["up"] = [self.normal_X]
                self.history_peaks["down"] = [-self.normal_X]

            elif self.last_date == self.time_X.date():
                if len(self.history_peaks["up"]) < self.num_over_threshold:
                    heapq.heappush(self.history_peaks["up"], self.normal_X)
                    # We use negative x to simulate a maxheap
                    heapq.heappush(self.history_peaks["down"], -self.normal_X)

                    # if len(self.history_peaks["up"]) == self.num_over_threshold:
                    self._update_oneside("up", init=True)
                    self._update_oneside("down", init=True)

                elif self.normal_X > self.local_init_threshold["up"]:
                    self._update_oneside("up")
                elif self.normal_X < self.local_init_threshold["down"]:
                    self._update_oneside("down")

        if self.index >= self.ignore_n:
            self.back_mean_window.append(X)
        return self

    def score(self, X: np.ndarray, timestamp=None) -> float:

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

        elif self.normal_X > self.global_init_threshold["up"]:
            side = "up"
            score = np.divide(
                self.normal_X - self.global_init_threshold[side],
                (
                    self.extreme_quantile[side]
                    - self.global_init_threshold[side]
                ),
                np.array(0.9),
                where=(
                    self.extreme_quantile[side]
                    - self.global_init_threshold[side]
                    != 0
                ),
            )
        elif self.normal_X < self.global_init_threshold["down"]:
            side = "down"
            score = np.divide(
                self.global_init_threshold[side] - self.normal_X,
                (
                    self.global_init_threshold[side]
                    - self.extreme_quantile[side]
                ),
                np.array(0.5),
                where=(
                    self.global_init_threshold[side]
                    - self.extreme_quantile[side]
                    != 0
                ),
            )
        else:
            score = 0.0

        return float(score)
