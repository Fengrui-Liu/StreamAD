#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-06 19:58:51
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
from StreamAD.model import xStreamDetector
from StreamAD.util import StreamGenerator
from sklearn.metrics import average_precision_score, roc_auc_score
import tqdm


CLASSES = [
    range(0, 1000),  # sparse benign cluster
    range(1000, 3000),  # dense benign cluster
    range(3000, 3050),  # clustered anomalies
    range(3050, 3075),  # sparse anomalies
    range(3076, 3082),  # local anomalies
    range(3075, 3076),
]  # single anomaly

k = 50
nchains = 40
depth = 10

X = np.loadtxt("../data/mllib/synDataNoisy.tsv")
y = np.array([0] * 3000 + [1] * 82)

data_X = pd.DataFrame(X)[2950:3080]
y = y[2950:3080]

stream = StreamGenerator(data_X, shuffle=False).iter_item()


def test_xStream():
    detector = xStreamDetector(n_components=k, n_chains=nchains)
    final_score = []
    for data, label in tqdm.tqdm(stream):

        score = detector.fit_score_partial(data)
        # print(score)
        final_score.append(-score[0])
    ap = average_precision_score(y, final_score)
    auc = roc_auc_score(y, final_score)
    print("xstream: AP =", ap, "AUC =", auc)


if __name__ == "__main__":
    test_xStream()