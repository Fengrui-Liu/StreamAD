#!/usr/bin/env python
# coding=utf-8
#
# Author: liufr
# Github: https://github.com/Fengrui-Liu
# LastEditTime: 2021-01-06 08:51:54
# Copyright 2021 liufr
# Description:
#


from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseDetector(ABC):
    """Abstract class for base detector

    Args:
        ABC (abstract): Abstract class
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit_partial(self, X: pd.Series, Y: pd.Series = None):
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
