from collections import deque
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from streamad.base import BaseDetector


class KNNDetector(BaseDetector):
    def __init__(self, k_neighbor: int = 5, **kwargs):
        """Univariate KNN-CAD model with mahalanobis distance :cite:`DBLP:journals/corr/BurnaevI16`.

        Args:
            k_neighbor (int, optional): The number of neighbors to cumulate distances. Defaults to 5.
        """
        super().__init__(data_type="univariate", **kwargs)
        self.window = deque(maxlen=int(np.sqrt(self.window_len)))
        self.buffer = deque(maxlen=self.window_len - self.window.maxlen)

        assert (
            k_neighbor < self.buffer.maxlen
        ), "k_neighbor must be less than the length of buffer"

        self.k = k_neighbor

    def fit(self, X: np.ndarray):

        self.window.append(X[0])

        if len(self.window) == self.window.maxlen:
            self.buffer.append(deepcopy(self.window))

        return self

    def score(self, X) -> float:

        window = deepcopy(self.window)
        window.pop()
        window.append(X[0])

        try:
            dist = cdist(np.array([window]), self.buffer, metric="mahalanobis")[
                0
            ]
        except:
            dist = cdist(
                np.array([window]),
                self.buffer,
                metric="mahalanobis",
                VI=np.linalg.pinv(self.buffer),
            )[0]
        score = np.sum(np.partition(np.array(dist), self.k + 1)[1 : self.k + 1])

        return float(score)
