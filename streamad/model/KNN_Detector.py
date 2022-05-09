from collections import deque
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from streamad.base import BaseDetector


class KNNDetector(BaseDetector):
    """Univariate KNN-CAD model with mahalanobis distance. :cite:`DBLP:journals/corr/BurnaevI16`. See `KNN-CAD <https://arxiv.org/abs/1608.04585>`_"""

    def __init__(
        self, window_len: int = 100, buffer_len=200, k_neighbor: int = 3
    ):
        """KNN anomaly detector with mahalanobis distance.

        Args:
            window_len (int, optional): The length of window. Defaults to 100.
            buffer_len (int, optional): The length of references. Defaults to 200.
            k_neighbor (int, optional): The number of neighbors to cumulate distances. Defaults to 3.
        """

        assert (
            k_neighbor < buffer_len
        ), "k_neighbor must be less than buffer_len"

        self.data_type = "univariate"
        self.buffer = deque(maxlen=buffer_len)
        self.window = deque(maxlen=window_len)
        self.scores = deque(maxlen=window_len)

        self.k = k_neighbor

    def fit(self, X: np.ndarray):
        """Record and analyse the current observation from the stream. Detector collect the init data firstly, and further score observation base on the observed data.

        Args:
            X (np.ndarray): Current observation.
        """

        self.window.append(X[0])

        if len(self.window) == self.window.maxlen:
            self.buffer.append(deepcopy(self.window))

        if len(self.buffer) == self.buffer.maxlen:
            if len(self.scores) == 0:
                all_dist = cdist(self.buffer, self.buffer, metric="mahalanobis")

                for dist in all_dist:
                    d = np.sum(
                        np.partition(np.array(dist), self.k + 1)[1 : self.k + 1]
                    )
                    self.scores.append(d)
            else:
                dist = cdist(
                    np.array([self.window]), self.buffer, metric="mahalanobis"
                )[0]
                d = np.sum(
                    np.partition(np.array(dist), self.k + 1)[1 : self.k + 1]
                )
                self.scores.append(d)

        return self

    def score(self, X) -> float:
        """Score the current observation. None for init period and float for the score of anomalies.

        Args:
            X (np.ndarray): Current observation.

        Returns:
            float: Anomaly probability.
        """

        if (
            len(self.window) + len(self.buffer)
            < self.window.maxlen + self.buffer.maxlen
        ):
            return None

        score = self.scores[-1]

        scores = np.array(self.scores)
        score_mean = np.mean(scores[:-1])
        score_std = np.std(scores[:-1])
        prob = (score - score_mean) / score_std
        if prob > 3:
            max_score = max(scores[:-1])
            prob = (score - score_mean) / (max_score - score_mean)
        else:
            return 0
        return abs(prob)
