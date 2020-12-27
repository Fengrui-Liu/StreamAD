#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-06 19:44:28
Copyright 2020 liufr
Description: StreamHash module for xStream
"""

import numpy as np
import pdb


class StreamhashProjector:
    def __init__(self, num_components, density=1 / 3.0):

        self.keys = np.arange(0, num_components, 1)
        self.constant = np.sqrt(1.0 / density) / np.sqrt(num_components)
        self.density = density
        self.n_components = num_components

    def fit_partial(self, X):
        """Fits particular (next) timestep's features to train the projector.

        Args:
            X (np.float array of shape (n_components,)): Input feature vector.

        Returns:
            object: self.
        """
        return self

    def transform_partial(self, X):
        """Projects particular (next) timestep's vector to (possibly) lower dimensional space.

        Args:
            X (np.float array of shape (num_features,)): Input feature vector.

        Returns:
            projected_X (np.float array of shape (num_components,)): Projected feature vector.
        """

        ndim = X.shape[0]

        feature_names = [str(i) for i in range(ndim)]

        R = np.array(
            [[self._hash_string(k, f) for f in feature_names] for k in self.keys]
        )

        Y = np.dot(X, R.T).squeeze()

        return Y

    def _hash_string(self, k, s):
        import mmh3

        hash_value = int(mmh3.hash(s, signed=False, seed=k)) / (2.0 ** 32 - 1)
        s = self.density
        if hash_value <= s / 2.0:
            return -1 * self.constant
        elif hash_value <= s:
            return self.constant
        else:
            return 0

    def fit_transform_partial(self, X):
        return self.fit_partial(X).transform_partial(X)