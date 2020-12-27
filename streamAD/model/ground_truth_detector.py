#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-29 10:26:00
Copyright 2020 liufr
Description: Ground truth detector
"""
from streamAD.base.BaseDetector import BaseDetector
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
