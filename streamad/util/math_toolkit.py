import collections
import math

import numpy as np
from collections import deque


class StreamStatistic:
    """Data statistics for the streaming data, with supporting max, min, sum, mean, sum of squares, var, std and standard scaler."""

    def __init__(self, is_global: bool = True, window_len: int = 10):
        self._is_uni = False
        self._is_global = is_global
        self._window = deque(maxlen=window_len)
        self._num_items = 0

        self._max = collections.defaultdict(lambda: -math.inf)
        self._min = collections.defaultdict(lambda: math.inf)
        self._sum = collections.defaultdict(float)
        self._mean = collections.defaultdict(float)
        self._sum_squares = collections.defaultdict(float)
        self._var = collections.defaultdict(float)
        self._std = collections.defaultdict(float)

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

            tmp = collections.defaultdict(float)

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
        Get max stattistic.
        """

        if self._is_global:
            result = [_ for _ in self._max.values()]
            if self._is_uni:
                return result[0]
            else:
                return np.array(result)
        else:
            if self._is_uni:
                return np.max(self._window, axis=0)[0]
            else:
                return np.max(self._window, axis=0)

    def get_min(self):
        """
        Get min stattistic.
        """

        if self._is_global:
            result = [_ for _ in self._min.values()]
            if self._is_uni:
                return result[0]
            else:
                return np.array(result)
        else:
            if self._is_uni:
                return np.min(self._window, axis=0)[0]
            else:
                return np.min(self._window, axis=0)

    def get_mean(self):
        """
        Get mean stattistic.
        """

        if self._is_global:
            result = [_ for _ in self._mean.values()]
            if self._is_uni:
                return result[0]
            else:
                return np.array(result)
        else:
            if self._is_uni:
                return np.mean(self._window, axis=0)[0]
            else:
                return np.mean(self._window, axis=0)

    def get_std(self):
        """
        Get max stattistic.
        """

        if self._is_global:
            result = [_ for _ in self._std.values()]
            if self._is_uni:
                return result[0]
            else:
                return np.array(result)
        else:
            if self._is_uni:
                return np.std(self._window, axis=0)[0]
            else:
                return np.std(self._window, axis=0)

    def get_sum(self):
        """
        Get sum stattistic.
        """

        if self._is_global:
            result = [_ for _ in self._sum.values()]
            if self._is_uni:
                return result[0]
            else:
                return np.array(result)
        else:
            if self._is_uni:
                return np.sum(self._window, axis=0)[0]
            else:
                return np.sum(self._window, axis=0)

    def get_var(self):
        """
        Get var stattistic.
        """

        if self._is_global:
            result = [_ for _ in self._var.values()]
            if self._is_uni:
                return result[0]
            else:
                return np.array(result)

        else:
            if self._is_uni:
                return np.var(self._window, axis=0)[0]
            else:
                return np.var(self._window, axis=0)
