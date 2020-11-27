#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-25 10:35:45
Copyright 2020 liufr
Description:
"""


import json
import os
import sys

from numpy.core.defchararray import mod

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../..")))

import numpy as np
import pytest
import streamingAD.util.window as window
from streamingAD.model import GroundTruthDetctor as gtd
from streamingAD.model import kNNAD
import pdb


np.random.seed(0)
n = 150
d = 2
X = np.random.randint(50, size=(n, d))
# Z = np.copy(X)
# Z[90:, :] = 1
Y = np.random.randint(2, size=n)

print(X)
# print(Y)
print("----")

windowsize = 9

data = window.tumble(X, windowsize, drop=False)
label = window.tumble(Y, windowsize, drop=False)

models_dict = {kNNAD: {"probationary_period": 20, "windowsize": windowsize}}


def test_window():
    # data = window.slide(X,11,drop=False)

    for point, label in zip(data, label):
        print(point, label)


# def test_ground_truth_model():
#     model = gtd()

#     for point, label in zip(data, label):
#         # print(point)
#         # print(label)

#         score = model.fit_partial(point, label).score_partial(point)

#         print("score：", score)


def run_models(data, detector, params):

    X_train = data

    # pdb.set_trace()
    Y_pred = detector.fit_score(X_train)
    print("score:", Y_pred)


def test_models():

    for model, params in models_dict.items():
        detector = model(**params)
        for point, point_label in zip(data, label):
            # print(point, point_label)
            run_models(point, detector, params)


if __name__ == "__main__":
    # test_window()
    # test_ground_truth_model()
    test_models()
