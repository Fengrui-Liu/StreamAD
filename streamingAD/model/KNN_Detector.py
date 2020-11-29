#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-29 21:21:55
Copyright 2020 liufr
Description: KNN-based anomaly detector
"""

import os
import pdb

import numpy as np
import pandas as pd
from scipy.spatial import distance
from streamingAD.base.BaseDetector import BaseDetector


class KNNDetector(BaseDetector):
    def __init__(
        self, observe_length: int, k_neighbor: int = 1, threshold: float = 0.9
    ) -> None:
        """KNN anomaly detector with mahalanobis distance.

        Args:
            observe_length (int): The history records used for the reference
            k_neighbor (int, optional): The number of neighbors to search for. Defaults to 1.
            threshold (float, optional): The threshold of anomaly probability. Defaults to 0.9.
        """
        self.training = []
        self.scores = []
        self.record_count = 0
        self.k = k_neighbor
        self.window_length = observe_length
        self.threshold = threshold
        self.sigma = np.diag(np.ones(self.window_length))
        self.feature_names = []

    def _ncm(self, item, item_in_array=False):

        arr = [distance.mahalanobis(x, item, self.sigma) for x in self.training]

        result = np.sum(
            np.partition(arr, self.k + item_in_array)[: self.k + item_in_array]
        )

        return result

    def fit_partial(self, X: pd.Series, y: pd.Series = None):
        """Record and anlyse one record from the stream

        Args:
            X (pd.Series): New item from the stream generator.
            y (pd.Series, optional): No need in this detector. Defaults to None.

        Raises:
            Exception: Input data shape is inconsistent.

        Returns:
            KNNDetector: self
        """
        if not self.feature_names:
            self.feature_names = X.index.tolist()
        elif self.feature_names != X.index.tolist():
            raise Exception(
                "Feature names change and operations abort. Please check the input data!"
            )

        self.training.append(X.values)
        self.record_count += 1

        if len(self.training) < self.window_length:
            return self

        ost = self.record_count % self.window_length
        if ost == 0 or ost == int(self.window_length / 2):
            cov = np.cov(self.training, rowvar=False)
            try:
                self.sigma = np.linalg.inv(cov)
            except np.linalg.linalg.LinAlgError:
                print("Singular Matrix at record", self.record_count)

        if len(self.scores) == 0:
            self.scores = [self._ncm(v, True) for v in self.training]

        new_item = X.values

        new_score = self._ncm(new_item)

        if self.record_count >= self.window_length * 2:
            self.training.pop(0)
            self.scores.pop(0)

        self.scores.append(new_score)

        return self

    def score_partial(self):
        """Score the last item.

        Returns:
            dict: A dict which contains anomaly tag according to threshold and corresponding probability.
        """
        if len(self.training) < self.window_length:
            return "Observating"

        new_score = self.scores[-1]

        result = (
            1.0 * len(np.where(np.array(self.scores) < new_score)[0]) / len(self.scores)
        )

        anomaly = False
        if result > self.threshold:
            anomaly = True

        return {"Anomaly": anomaly, "Anomaly Probability": result}
