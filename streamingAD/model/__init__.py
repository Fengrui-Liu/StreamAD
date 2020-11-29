#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-29 21:06:44
Copyright 2020 liufr
Description: Models collection
"""

from .ground_truth_detector import GroundTruthDetctor
from .KNN_Detector import KNNDetector

__all__ = ["GroundTruthDetctor", "KNNDetector"]
