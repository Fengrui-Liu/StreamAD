#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-25 10:34:15
Copyright 2020 liufr
Description: KNN-based anomaly detector
"""

import os
import pdb
from types import new_class
from numpy.core.records import array
from numpy.lib.function_base import diff
from streamingAD.core.BaseDetector import BaseDetector
import numpy as np


class kNNAD(BaseDetector):
    def __init__(
        self, probationary_period, windowsize, k_neighbor=8, threshold=0.9
    ) -> None:
        self.buf = []
        self.training = []
        self.calibration = []
        self.scores = []
        self.record_count = 0
        self.pred = -1
        self.k = k_neighbor
        self.to_init = True
        self.dim = windowsize
        self.threshold = threshold
        self.sigma = np.diag(np.ones(self.dim))
        self.probationary_period = probationary_period

    def _metric(self, a, b):
        # pdb.set_trace()
        diff = a - np.array(b)
        s_result = np.dot(diff.T, self.sigma)
        return np.dot(s_result, diff)

    def _ncm(self, item, item_in_array=False):

        arr = [self._metric(x, item) for x in self.training]

        result = np.sum(
            np.partition(arr, self.k + item_in_array)[: self.k + item_in_array]
        )

        return result

    def fit_partial(self, X, Y=None):

        # if self.to_init:
        #     self.to_init = False

        self.buf.append(X[0])
        self.record_count += 1

        if len(self.buf) < self.dim:
            return self
        # pdb.set_trace()
        new_item = self.buf[-self.dim :]

        if self.record_count < self.probationary_period:

            self.training.append(new_item)
        else:

            ost = self.record_count % self.probationary_period
            if ost == 0 or ost == int(self.probationary_period / 2):
                try:
                    # pdb.set_trace()
                    tra_a = np.array(self.training).swapaxes(1, 0)
                    tra_b = np.array(self.training)
                    temp = np.dot(tra_a, tra_b)
                    self.sigma = np.linalg.inv(temp)
                    print("success")
                except np.linalg.linalg.LinAlgError:
                    print("Singular Matrix at record", self.record_count)

            if len(self.scores) == 0:
                self.scores = [self._ncm(v, True) for v in self.training]

            new_score = self._ncm(new_item)

            if self.record_count >= 2 * self.probationary_period:
                self.training.pop(0)
                self.training.append(self.calibration.pop(0))

            self.scores.pop(0)
            self.calibration.append(new_item)
            self.scores.append(new_score)

        return self

    def score_partial(self, X):
        if len(self.buf) < self.dim or self.record_count < self.probationary_period:
            print(X)
            return 0.0
        new_item = self.buf[-self.dim :]
        pdb.set_trace()
        new_score = self._ncm(new_item)

        result = (
            1.0 * len(np.where(np.array(self.scores) < new_score)[0]) / len(self.scores)
        )
        if self.pred > 0:
            self.pred = -1
            return 0.5
        elif result >= 0.996:
            self.pred = int(self.probationary_period / 5)

        return result