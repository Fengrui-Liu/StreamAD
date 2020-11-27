#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-24 16:48:33
Copyright 2020 liufr
Description: Slide and tumble window for data.
"""


from collections import deque
from typing import Iterator
import numpy as np
import pdb
import math
import warnings


def slide(sequence, window_size=5, drop=False):
    """[Function of slide window]

    Args:
        sequence ([numpy.array]): [Input data, support for multivative data]
        window_size (int, optional): [Slide window size]. Defaults to 5.
        drop (bool, optional): [Drop the latest part, which cannot fill the whole window]. Defaults to False.

    Yields:
        [numpy.array]: [numpy array with window size, None for drop==True and sequence is smaller than window size]
    """
    iterator = iter(sequence)
    init = (next(iterator) for _ in range(min(len(sequence), window_size)))
    window = deque(init, maxlen=window_size)

    if len(window) < window_size and drop == True:
        warnings.warn("Sequence is smaller than window size")
        yield None
    else:
        yield np.asarray(window)
        for item in iterator:
            window.append(item)
            yield np.asarray(window)


def tumble(sequence, window_size=5, drop=False):
    """[Function of tumble window]

    Args:
        sequence ([numpy.array]): [Input data, support for multivative data]
        window_size (int, optional): [Slide window size]. Defaults to 5.
        drop (bool, optional): [Drop the latest part, which cannot fill the whole window]. Defaults to False.

    Yields:
        [numpy.array]: [numpy array with window size, None for drop==True and sequence is smaller than window size]
    """

    iterator = iter(sequence)
    window_count = math.floor(len(sequence) / window_size)
    latest_sequence_size = len(sequence) % window_size

    for _ in range(window_count):
        init = (next(iterator) for _ in range(window_size))
        window = deque(init, maxlen=window_size)
        yield np.asarray(window)

    if drop == True:
        yield None
    else:
        init = (next(iterator) for _ in range(latest_sequence_size))
        window = deque(init, maxlen=window_size)
        yield np.asarray(window)
