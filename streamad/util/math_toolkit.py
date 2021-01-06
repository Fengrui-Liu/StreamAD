#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-23 16:32:10
Copyright 2020 liufr
Description: Mathtookit for stream statistic
"""
import math
import collections
import numpy as np


class StreamStatistic:
    def __init__(self):
        """Statistic for stream
        We support max, min, sum, mean, sum of squares, var, std and standard scaler for streams
        """
        self.num_items = 0
        self.max = collections.defaultdict(lambda: -math.inf)
        self.min = collections.defaultdict(lambda: math.inf)
        self.sum = collections.defaultdict(float)
        self.mean = collections.defaultdict(float)
        self.sum_squares = collections.defaultdict(float)
        self.var = collections.defaultdict(float)
        self.std = collections.defaultdict(float)

    def update(self, num):
        """Update a pd.Series to stream

        Args:
            num ([pd.Series]): An item from StreamGenerator
        """
        self.num_items += 1

        tmp = collections.defaultdict(float)

        for index, item in num.items():
            # print(index, "---", item)
            self.max[index] = self.max[index] if self.max[index] > item else item
            self.min[index] = self.min[index] if self.min[index] < item else item
            self.sum[index] += num[index]
            old_mean = self.mean[index]
            tmp[index] = item - self.mean[index]
            self.mean[index] = self.sum[index] / self.num_items
            self.sum_squares[index] += (num[index] - old_mean) * (
                num[index] - self.mean[index]
            )
            self.var[index] = self.sum_squares[index] / self.num_items
            self.std[index] = math.sqrt(self.var[index])
        return self

    def standard_scaler(self, num):
        """Standard scaler for current item of stream

        Args:
            num (pd.Series): Update and standard scale for item withe the same item

        Returns:
            pd.Series: Results after standard scaling
        """
        stand_result = collections.defaultdict(float)
        for index, item in num.items():
            stand_result[index] = (
                np.divide((item - self.mean[index]), self.std[index])
                if self.std[index] != 0
                else 0
            )

        return pd.Series(stand_result)

    def update_standard_scaler(self, num):

        return self.update(num).standard_scaler(num)