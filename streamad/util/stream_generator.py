#!/usr/bin/env python
# coding=utf-8
#
# Author: liufr
# Github: https://github.com/Fengrui-Liu
# LastEditTime: 2021-01-09 20:35:38
# Copyright 2021 liufr
# Description:
#

from typing import Generator, Optional, Union

import numpy as np
import pandas as pd


class StreamGenerator:
    """Load static dataset and generate observation once a time."""

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        features: list = None,
        shuffle: bool = False,
    ):
        """Initialize a StreamGenerator.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Origin static dataset.
            features (list, optional): Selected features from pd.DataFrame, None for all features and ingore for np.ndarray. Defaults to None.
            shuffle (bool, optional): Reorder the data. Defaults to False.
            y (Optional[Union[pd.Series, np.ndarray]]): Label of origin static dataset. Defaults to False.

        Raises:
            TypeError: Unexpected input data type.
        """
        if y is not None:
            assert len(X) == len(y)

        if isinstance(X, np.ndarray):
            self.X = X
            self.y = [None] * len(X) if y is None else y
        elif isinstance(X, pd.DataFrame):
            self.X = X.to_numpy() if features == None else X[features].to_numpy()
            self.y = [None] * len(X) if y is None else y.to_numpy()
        else:
            raise TypeError(
                "Unexpected input data type, except np.ndarray or pd.DataFrame"
            )
        self.features = features
        self.index = list(range(len(X)))
        if shuffle:
            np.random.shuffle(self.index)

    def iter_item(self) -> Generator:
        """Iterate item once a time from the dataset.

        Yields:
            Generator: One observation and corresponding label from dataset.
        """

        for i in self.index:
            yield self.X[i], self.y[i]

    def get_features(self) -> list:
        """Get the selected features of current StreamGenerator

        Returns:
            list: Selected features
        """
        return self.features
