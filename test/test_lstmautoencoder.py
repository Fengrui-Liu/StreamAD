#!/usr/bin/env python
# coding=utf-8
#
# Author: liufr
# Github: https://github.com/Fengrui-Liu
# LastEditTime: 2021-01-08 16:20:56
# Copyright 2021 liufr
# Description:
#


import os

import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))
from StreamAD.model import LSTMDetector
from StreamAD.util import StreamGenerator
import pandas as pd

df = pd.read_csv("./data/mllib/FOREST.csv", header=None)
data = df.iloc[:, 0:7]
# normal:0 , anomaly:1
label = df.iloc[:, -1]
# print(data.head)
# print(label.head)

stream = StreamGenerator(data, label).iter_item()


def test_lstm():
    detector = LSTMDetector(trainset_size=2000, window_size=20)
    for data, label in stream:
        a = detector.fit_score_partial(data)

        if a != None:
            print(a)


if __name__ == "__main__":
    dataPath = "../data/mllib/FOREST.csv"
    modelSavePath = "./tmp_model"
    test_lstm()