#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-16 19:47:13
Copyright 2020 liufr
Description: Models collection
"""


from .multivariate.KNN_Detector import KNNDetector
from .multivariate.xStream_Detector import xStreamDetector
from .univariate.spot_Detector import SpotDetector


__all__ = ["KNNDetector", "xStreamDetector", "SpotDetector"]
