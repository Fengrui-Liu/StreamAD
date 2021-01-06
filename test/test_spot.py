#!/usr/bin/env python
# coding=utf-8
#
# Author: liufr
# Github: https://github.com/Fengrui-Liu
# LastEditTime: 2021-01-05 21:32:16
# Copyright 2021 liufr
# Description:
#


import os
import sys
import StreamAD.base.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from StreamAD.model import SpotDetector
from StreamAD.util import StreamGenerator

matplotlib.use("TkAgg")

f = "./data/mllib/physics.dat"
r = open(f, "r").read().split(",")
data_X = np.array(list(map(float, r)))
data_X = data_X[:4000]
stream = StreamGenerator(data_X, shuffle=False).iter_item()

win = 100
initlength = 1000
detector = SpotDetector(proba=1e-3, window_size=win, init_length=initlength)

index = 0
alarm = []
for data, label in stream:
    index += 1
    result = detector.fit_score_partial(data)
    if result == 1:
        alarm.append(index)
        # print(result)
        print("anomaly")


# print(alarm)


def plot(run_results, with_alarm=True):
    deep_saffron = "#FF9933"
    air_force_blue = "#5D8AA8"

    x = range(data_X.size)
    K = run_results.keys()

    (ts_fig,) = plt.plot(x, data_X, color=air_force_blue)
    fig = [ts_fig]

    if "upper_thresholds" in K:
        thup = run_results["upper_thresholds"]
        (uth_fig,) = plt.plot(
            x[initlength:], thup, color=deep_saffron, lw=2, ls="dashed"
        )
        fig.append(uth_fig)

    if "lower_thresholds" in K:
        thdown = run_results["lower_thresholds"]
        (lth_fig,) = plt.plot(
            x[initlength:], thdown, color=deep_saffron, lw=2, ls="dashed"
        )
        fig.append(lth_fig)

    if with_alarm and ("alarms" in K):
        alarm = np.array(run_results["alarms"]) - 1
        if len(alarm) > 0:
            al_fig = plt.scatter(alarm, data_X[alarm], color="red")
            fig.append(al_fig)

    plt.xlim((0, data_X.size))
    plt.show()
    return fig


if __name__ == "__main__":
    plot(
        {
            "upper_thresholds": detector.thup,
            "lower_thresholds": detector.thdown,
            "alarms": alarm,
        }
    )
