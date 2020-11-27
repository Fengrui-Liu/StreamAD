#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-27 17:35:55
Copyright 2020 liufr
Description:
"""
import math


class StreamStatistic:
    def __init__(self):

        self.num_items = 0
        self.max = -math.inf
        self.min = math.inf
        self.sum = 0.0
        self.mean = 0.0

        self.var = 0.0

    def update(self, num):
        self.num_items += 1
        self.max = self.max if self.max > num else num
        self.min = self.min if self.min < num else num
        self.sum += num
        old_mean = self.mean
        self.mean = self.sum / self.num_items
        self.var = ((num - old_mean) * (num - self.mean)) / self.num_items
