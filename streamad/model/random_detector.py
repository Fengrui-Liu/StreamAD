#!/usr/bin/env python
# coding=utf-8
#
# Author: liufr
# Github: https://github.com/Fengrui-Liu
# LastEditTime: 2021-01-10 19:20:30
# Copyright 2021 liufr
# Description:
#

import random
from typing import Optional, Union

import numpy as np
import pandas as pd
from streamad.base import BaseDetector


class RandomDetector(BaseDetector):
    """Return random anomaly score. A minimum score for benchmark."""

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> None:
        """Detector fit current observation from StreamGenerator.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The data value of current observation from StreamGenerator.
        """
        pass

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """Abstract method: Detector predict the probability of anomaly for current observation form StreamGenerator.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The data value of current observation from StreamGenerator.

        Returns:
            float: Anomaly probability.
        """

        return random.random()
