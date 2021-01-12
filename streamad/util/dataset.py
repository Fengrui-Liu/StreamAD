#!/usr/bin/env python
# coding=utf-8
#
# Author: liufr
# Github: https://github.com/Fengrui-Liu
# LastEditTime: 2021-01-11 14:38:40
# Copyright 2021 liufr
# Description:
#

from os.path import dirname, join
import numpy as np
import pandas as pd


class MultivariateDS(object):
    """
    Load multivariate dataset.
    synthetic data point classes
    CLASSES = [
    range(0, 1000),  # sparse benign cluster
    range(1000, 3000),  # dense benign cluster
    range(3000, 3050),  # clustered anomalies
    range(3050, 3075),  # sparse anomalies
    range(3076, 3082),  # local anomalies
    range(3075, 3076),# single anomaly
    ]
    """

    def __init__(self) -> None:
        super().__init__()
        module_path = dirname(__file__)
        data_path = join(module_path, "../", "data", "multiDS.csv")
        self.data = np.loadtxt(data_path)
        self.label = np.array([0] * 3000 + [1] * 82)


class UnivariateDS(object):
    """
    Load univariate dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        module_path = dirname(__file__)
        data_path = join(module_path, "../", "data", "uniDS.csv")
        data = pd.read_csv(data_path)
        self.data = data["value"].to_numpy()
        self.label = data["is_anomaly"].to_numpy()
