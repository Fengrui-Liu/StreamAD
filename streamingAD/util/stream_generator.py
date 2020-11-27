#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-27 16:03:10
Copyright 2020 liufr
Description: Iterate item of pandas, numpy array, list
"""

import random
from operator import le
from typing import Generator, Iterable, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class StreamGenerator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        shuffle: bool = False,
    ):
        """Init stream generator

        Args:
            X (np.ndarray or pd.Dataframe): Origin dataset
            y (np.ndarray or pd.Dataframe, optional): Origin label. Defaults to None.
            shuffle (bool, optional): Shuffle or not. Defaults to False.
        """

        if y is not None:
            assert len(X) == len(y)
        self.X = X
        self.y = y
        self.index = list(range(len(X)))
        if shuffle:
            np.random.shuffle(self.index)

    def iter_item(self) -> Generator:
        """Iterate item in dataset

        Raises:
            Exception: Unsupported datatype

        Yields:
            Generator: One item of dataset and labels
        """
        if isinstance(self.X, (np.ndarray)):
            yield from self._iter_array()
        elif isinstance(self.X, (pd.DataFrame)):
            self.X = self.X.to_numpy()
            self.y = self.y.to_numpy() if self.y is not None else None
            yield from self._iter_array()
        else:
            raise Exception(
                "Unexcepted type, only accept numpy.ndarray or pandas.dataframe"
            )

    def _iter_array(self):

        if self.y is None:
            for i in self.index:
                yield self.X[i], None
        else:
            for i in self.index:
                yield self.X[i], self.y[i]
