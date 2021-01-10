#!/usr/bin/env python
# coding=utf-8
#
# Author: liufr
# Github: https://github.com/Fengrui-Liu
# LastEditTime: 2021-01-09 19:24:46
# Copyright 2021 liufr
# Description:
#
import collections
import math
from typing import Union

import numpy as np


class StreamStatistic:
    """Data statistics for the streaming data."""

    def __init__(self):
        """Statistic for stream data
        We support max, min, sum, mean, sum of squares, var, std and standard scaler for streaming data.
        """
        self.num_items = 0

        self.max = collections.defaultdict(lambda: -math.inf)
        self.min = collections.defaultdict(lambda: math.inf)
        self.sum = collections.defaultdict(float)
        self.mean = collections.defaultdict(float)
        self.sum_squares = collections.defaultdict(float)
        self.var = collections.defaultdict(float)
        self.std = collections.defaultdict(float)

    def update(self, X: np.ndarray):
        """Update a pd.Series to stream

        Args:
            X (np.ndarray): An item from StreamGenerator

        """

        self.num_items += 1

        tmp = collections.defaultdict(float)

        for index, item in enumerate(X):
            self.max[index] = self.max[index] if self.max[index] > item else item
            self.min[index] = self.min[index] if self.min[index] < item else item
            self.sum[index] += X[index]
            old_mean = self.mean[index]
            tmp[index] = item - self.mean[index]
            self.mean[index] = self.sum[index] / self.num_items
            self.sum_squares[index] += (X[index] - old_mean) * (
                X[index] - self.mean[index]
            )
            self.var[index] = self.sum_squares[index] / self.num_items
            self.std[index] = math.sqrt(self.var[index])
