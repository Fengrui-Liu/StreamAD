#!/usr/bin/env python
# coding=utf-8
#
# Author: liufr
# Github: https://github.com/Fengrui-Liu
# LastEditTime: 2021-01-10 19:10:32
# Copyright 2021 liufr
# Description:
#


from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class BaseDetector(ABC):
    """Abstract class for BaseDetector, supporting for customize detector."""

    def __init__(self):
        """Initialization BaseDetector"""
        pass

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> None:
        """Detector fit current observation from StreamGenerator.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The data value of current observation from StreamGenerator.
        """
        pass

    @abstractmethod
    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """Detector score the probability of anomaly for current observation form StreamGenerator.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The data value of current observation from StreamGenerator.

        Returns:
            float: Anomaly probability. 1.0 for anomaly and 0.0 for normal.
        """

        return 0.0

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5
    ) -> int:
        """Detector predict the label of current observation form StreamGenerator.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The data value of current observation from StreamGenerator.
            threshold (float, optional): Threshold for labeling from data's score. Defaults to 0.5.

        Returns:
            int: Data Label. 1 for anomaly and 0 for normal.
        """
        if threshold >= 1 or threshold <= 0:
            raise ValueError(
                "Invalid threshold %f, please select a threshold between (0,1)"
            )
        probability = self.score(X)
        if probability >= threshold:
            return 1
        return 0
