#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-23 11:16:51
Copyright 2020 liufr
Description: Ground truth detector
"""
from streamingAD.core.BaseDetector import BaseDetector
from collections import deque


class GroundTruthDetctor(BaseDetector):
    def __init__(self):
        self.labels = deque()

    def fit_partial(self, X, Y):
        self.labels = deque([Y])

        return self

    def score_partial(self, X):
        score = self.labels.popleft()
        return score
