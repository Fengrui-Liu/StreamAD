#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-10 15:13:09
Copyright 2020 liufr
Description:
"""

import os

import sys

from numpy.core.defchararray import mod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))
from streamingAD.model import LSTMDetector
from streamingAD.util import StreamGenerator
import pandas as pd

df = pd.read_csv("../data/mllib/FOREST.csv", header=None)
data = df.iloc[:, 0:7]
# normal:0 , anomaly:1
label = df.iloc[:, -1]
# print(data.head)
# print(label.head)

strem = StreamGenerator(data, label).iter_item()


def test_lstm():
    detector = LSTMDetector()
    for data, label in strem:
        a = detector.fit_partial(data, label)
        if a == True:
            break


if __name__ == "__main__":
    dataPath = "../data/mllib/FOREST.csv"
    modelSavePath = "./tmp_model"
    test_lstm()