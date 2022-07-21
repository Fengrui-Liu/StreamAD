import math

import numpy as np
from collections import deque, defaultdict


class StreamStatistic:
    """Data statistics for the streaming data, with supporting max, min, sum, mean, sum of squares, var, std and standard scaler."""

    def __init__(self, is_global: bool = True, window_len: int = 10):
        """Statistics for the streaming data, with supporting max, min, sum, mean, sum of squares, var, std and standard scaler.

        Args:
            is_global (bool, optional): For whole stream or a windowed stream. Defaults to True.
            window_len (int, optional): Rolloing window length. Only works when is_global is False. Defaults to 10.
        """
        self._is_uni = False
        self._is_global = is_global
        self._window = deque(maxlen=window_len)
        self._num_items = 0

        self._max = defaultdict(lambda: -math.inf)
        self._min = defaultdict(lambda: math.inf)
        self._sum = defaultdict(float)
        self._mean = defaultdict(float)
        self._sum_squares = defaultdict(float)
        self._var = defaultdict(float)
        self._std = defaultdict(float)

    def update(self, X: np.ndarray):
        """Update a pd.Series to stream

        Args:
            X (np.ndarray): An item from StreamGenerator

        """

        self._num_items += 1

        if isinstance(X, int) or isinstance(X, float):
            X = np.array([X])
            self._is_uni = True
        elif isinstance(X, np.ndarray):
            X = np.array([X]).flatten()
            if len(X) == 1:
                self._is_uni = True
            else:
                self._is_uni = False
        else:
            raise NotImplementedError("Only support int, float and np.ndarray")

        if self._is_global:

            tmp = defaultdict(float)

            for index, item in enumerate(X):
                self._max[index] = (
                    self._max[index] if self._max[index] > item else item
                )
                self._min[index] = (
                    self._min[index] if self._min[index] < item else item
                )
                self._sum[index] += X[index]
                old_mean = self._mean[index]
                tmp[index] = item - self._mean[index]
                self._mean[index] = self._sum[index] / self._num_items
                self._sum_squares[index] += (X[index] - old_mean) * (
                    X[index] - self._mean[index]
                )
                self._var[index] = self._sum_squares[index] / self._num_items
                self._std[index] = math.sqrt(self._var[index])
        else:
            self._window.append(X)

    def get_max(self):
        """
        Get max statistic.
        """

        if self._is_global:
            result = [_ for _ in self._max.values()]
        else:
            result = np.max(self._window, axis=0)

        return result[0] if self._is_uni else np.array(result)

    def get_min(self):
        """
        Get min statistic.
        """

        if self._is_global:
            result = [_ for _ in self._min.values()]
        else:
            result = np.min(self._window, axis=0)

        return result[0] if self._is_uni else np.array(result)

    def get_mean(self):
        """
        Get mean statistic.
        """

        if self._is_global:
            result = [_ for _ in self._mean.values()]
        else:
            result = np.mean(self._window, axis=0)

        return result[0] if self._is_uni else np.array(result)

    def get_std(self):
        """
        Get max statistic.
        """

        if self._is_global:
            result = [_ for _ in self._std.values()]
        else:
            result = np.std(self._window, axis=0)

        return result[0] if self._is_uni else np.array(result)

    def get_sum(self):
        """
        Get sum statistic.
        """

        if self._is_global:
            result = [_ for _ in self._sum.values()]
        else:
            result = np.sum(self._window, axis=0)

        return result[0] if self._is_uni else np.array(result)

    def get_var(self):
        """
        Get var statistic.
        """

        if self._is_global:
            result = [_ for _ in self._var.values()]
        else:
            result = np.var(self._window, axis=0)

        return result[0] if self._is_uni else np.array(result)
