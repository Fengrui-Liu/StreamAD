from collections import deque

import numpy as np
from streamad.base import BaseDetector


class LodaDetector(BaseDetector):
    def __init__(
        self, window_len: int = 100, random_cuts_num: int = 100,
    ):
        """Multivariate LODA Detector :cite:`DBLP:journals/ml/Pevny16`.

        Args:
            window_len (int, optional): The length of window. Defaults to 100.
            random_cuts_num (int, optional): The number of random experiments. Defaults to 100.
        """
        super().__init__()

        self.window_len = window_len
        self.window = deque(maxlen=window_len)

        self.random_cuts_num = random_cuts_num
        self.bins_num = int(1 * (window_len ** 1) * (np.log(window_len) ** -1))
        self._weights = np.ones(random_cuts_num) / random_cuts_num
        self.components_num = None
        self.nonzero_components_num = None
        self.zero_components_num = None
        self._projections = None
        self._histograms = None
        self._limits = None

    def fit(self, X: np.ndarray):
        self.window.append(X)
        if self.index == 0:
            self.components_num = len(X)
            self.nonzero_components_num = int(np.sqrt(self.components_num))
            self.zero_components_num = (
                self.components_num - self.nonzero_components_num
            )

        elif len(self.window) == self.window.maxlen:
            self._projections = np.random.randn(
                self.random_cuts_num, self.components_num
            )
            self._histograms = np.zeros([self.random_cuts_num, self.bins_num])
            self._limits = np.zeros([self.random_cuts_num, self.bins_num + 1])

            for i in range(self.random_cuts_num):
                rands = np.random.permutation(self.components_num)[
                    : self.zero_components_num
                ]
                self._projections[i, rands] = 0.0
                projected_data = self._projections[i, :].dot(
                    np.array(self.window).T
                )
                self._histograms[i, :], self._limits[i, :] = np.histogram(
                    projected_data, bins=self.bins_num, density=False
                )
                self._histograms[i, :] += 1e-12
                self._histograms[i, :] /= np.sum(self._histograms[i, :])

        return self

    def score(self, X: np.ndarray):

        score = 0

        for i in range(self.random_cuts_num):
            projected_data = self._projections[i, :].dot(np.array(X).T)
            inds = np.searchsorted(
                self._limits[i, : self.bins_num - 1],
                projected_data,
                side="left",
            )
            score += -self._weights[i] * np.log(self._histograms[i, inds])

        score = score / self.random_cuts_num
        return float(score)
