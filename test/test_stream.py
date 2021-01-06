#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-03 19:09:48
Copyright 2020 liufr
Description:
"""


import os
import sys

from numpy.core.defchararray import mod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

import pdb

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine
from StreamAD.util import StreamGenerator, StreamStatistic
from StreamAD.model import KNNDetector, xStreamDetector

dataset = load_wine()
X = dataset.data
Y = dataset.target

data_X = pd.DataFrame(X, columns=[dataset.feature_names])
data_Y = pd.DataFrame(Y, columns=["target"])
# data = pd.concat([data_X,data_Y],axis=1)

stream = StreamGenerator(
    data_X, data_Y, shuffle=False, feature_names=dataset.feature_names
).iter_item()

# print(data_X)


def test_generator():

    for data, label in stream:

        print(data, label)


def test_math_toolkit():
    stat = StreamStatistic()
    a = [
        {"x": 10.557, "y": 8.100},
        {"x": 9.100, "y": 8.892},
        {"x": 10.945, "y": 10.706},
        {"x": 11.568, "y": 8.347},
        {"x": 9.687, "y": 8.119},
        {"x": 8.874, "y": 10.021},
    ]
    b = [[-2.1, -1, 4.3], [3, 1.1, 0.12]]
    b = np.array(b).T
    print(b)
    b = pd.DataFrame(b)
    b = StreamGenerator(b, shuffle=Falsekurtosis).iter_item()

    # for item, label in stream:
    for item, label in b:
        print("???", item)
        stat.update(item)
        # print(stat.max)

        print(stat.cov)
    # print(np.cov(data_X, rowvar=True))
    # print(stat.max["ash"])


def test_knncad():
    detector = KNNDetector(observation_length=-1, k_neighbor=7)

    for data, label in stream:
        y_pred = detector.fit_score(data)
        print(y_pred)


def test_xStream():
    detector = xStreamDetector()


if __name__ == "__main__":
    # test_generator()
    test_math_toolkit()
    # test_knncad()
