#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-28 20:37:49
Copyright 2020 liufr
Description:
"""
import math
import collections
import numpy as np

# class StreamStatistic:
#     def __init__(self, dim=1):

#         self.num_items = 0
#         self.max = [-math.inf] * dim
#         self.min = [math.inf] * dim
#         self.sum = [0.0] * dim
#         self.mean = [0.0] * dim
#         self.sum_squares = [0.0] * dim
#         self.var = [0.0] * dim
#         self.std = [0.0] * dim
#         self.dim = dim

#     def update(self, num):
#         #
#         self.num_items += 1
#         for index, item in enumerate(num):
#             self.max[index] = self.max[index] if self.max[index] > item else item
#             self.min[index] = self.min[index] if self.min[index] < item else item
#             self.sum[index] += num[index]
#             old_mean = self.mean[index]
#             self.mean[index] = self.sum[index] / self.num_items
#             self.sum_squares[index] += (num[index] - old_mean) * (
#                 num[index] - self.mean[index]
#             )
#             self.var[index] = self.sum_squares[index] / self.num_items
#             self.std[index] = math.sqrt(self.var[index])


class StreamStatistic:
    def __init__(self):

        self.num_items = 0
        self.max = collections.defaultdict(lambda: -math.inf)
        self.min = collections.defaultdict(lambda: math.inf)
        self.sum = collections.defaultdict(float)
        self.mean = collections.defaultdict(float)
        self.sum_squares = collections.defaultdict(float)
        self.var = collections.defaultdict(float)
        self.std = collections.defaultdict(float)

    def update(self, num):
        self.num_items += 1
        for index, item in num.items():
            # print(index, "---", item)
            self.max[index] = self.max[index] if self.max[index] > item else item
            self.min[index] = self.min[index] if self.min[index] < item else item
            self.sum[index] += num[index]
            old_mean = self.mean[index]
            self.mean[index] = self.sum[index] / self.num_items
            self.sum_squares[index] += (num[index] - old_mean) * (
                num[index] - self.mean[index]
            )
            self.var[index] = self.sum_squares[index] / self.num_items
            self.std[index] = math.sqrt(self.var[index])

    def standard_scaler(self, num):

        stand_result = collections.defaultdict(float)
        for index, item in num.items():
            stand_result[index] = np.divide(
                (item - self.mean[index]), self.std[index], where=self.std[index] != 0
            )

        return stand_result
