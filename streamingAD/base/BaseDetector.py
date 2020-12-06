#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-04 09:55:09
Copyright 2020 liufr
Description: Base class of all detectors
"""

from abc import ABC, abstractmethod
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

    def fit_score_partial(self, X, Y=None):

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

        if Y == None:
            Y = np.empty(len(X))

        pred_result = self.fit_score_partial(X, Y)

        return pred_result
