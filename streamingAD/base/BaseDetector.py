#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-24 16:47:17
Copyright 2020 liufr
Description: Base class of all detectors
"""

from abc import ABC, abstractmethod
import pdb
from sklearn.utils.validation import check_is_fitted
import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit_partial(self, X, Y=None):
        pass

    @abstractmethod
    def score_partial(self, X):
        pass

    def fit_score_partial(self, X, Y):

        return self.fit_partial(X, Y).score_partial(X)

    def fit(self, X, Y=None):

        for x, y in zip(X, Y):
            self.fit_partial(x, y)

        return self

    def score(self, X):

        y_pred = list()

        for x in X:
            y_pred.append(self.score_partial(x))

        return np.array(y_pred)

    def fit_score(self, X, Y=None):
        y_pred = list()

        if Y == None:
            Y = np.empty(len(X))

        for x, y in zip(X, Y):
            pred_result = self.fit_score_partial(x, y)
            y_pred.append(pred_result)

        return np.array(y_pred)
