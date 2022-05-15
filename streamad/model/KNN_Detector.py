from collections import deque
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from streamad.base import BaseDetector
from streamad.util import StreamStatistic


class KNNDetector(BaseDetector):
    def __init__(self, window_len: int = 10, init_len=150, k_neighbor: int = 3):
        """Univariate KNN-CAD model with mahalanobis distance :cite:`DBLP:journals/corr/BurnaevI16`.

        Args:
            window_len (int, optional): The length of window. Defaults to 10.
            init_len (int, optional): The length of references. Defaults to 150.
            k_neighbor (int, optional): The number of neighbors to cumulate distances. Defaults to 3.
        """

        assert k_neighbor < init_len, "k_neighbor must be less than init_len"

        self.data_type = "univariate"
        self.buffer = deque(maxlen=init_len)
        self.window = deque(maxlen=window_len)
        # self.scores = deque(maxlen=window_len)
        self.prob = 0
        self.score_stats = StreamStatistic()

        self.k = k_neighbor
        self.stats = StreamStatistic()

    def fit(self, X: np.ndarray):

        self.window.append(X[0])

        if len(self.window) == self.window.maxlen:
            self.buffer.append(deepcopy(self.window))

        if len(self.buffer) == self.buffer.maxlen:
            if self.score_stats._num_items == 0:
                all_dist = cdist(self.buffer, self.buffer, metric="mahalanobis")

                for dist in all_dist:
                    self.prob = np.sum(
                        np.partition(np.array(dist), self.k + 1)[1 : self.k + 1]
                    )
                    self.score_stats.update(self.prob)
            else:
                dist = cdist(
                    np.array([self.window]), self.buffer, metric="mahalanobis"
                )[0]
                self.prob = np.sum(
                    np.partition(np.array(dist), self.k + 1)[1 : self.k + 1]
                )
                self.score_stats.update(self.prob)

        return self

    def score(self, X) -> float:

        if (
            len(self.window) + len(self.buffer)
            < self.window.maxlen + self.buffer.maxlen
        ):
            return None

        score_mean = self.score_stats.get_mean()
        score_std = self.score_stats.get_std()
        z_score = (self.prob - score_mean) / score_std
        if z_score > 3:
            max_score = self.score_stats.get_max()
            self.prob = (self.prob - score_mean) / (max_score - score_mean)
        else:
            return 0
        return abs(self.prob)
