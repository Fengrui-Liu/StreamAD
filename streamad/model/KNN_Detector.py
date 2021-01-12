import pdb
from warnings import WarningMessage
import warnings
import numpy as np
from numpy.core.defchararray import array
import pandas as pd
from scipy.spatial import distance
from streamad.base import BaseDetector


class KNNDetector(BaseDetector):
    """Univariate KNN-CAD model with mahalanobis distance. :cite:`DBLP:journals/corr/BurnaevI16`. See `KNN-CAD <https://arxiv.org/abs/1608.04585>`_"""

    def __init__(self, init_len: int = -1, k_neighbor: int = 25, all_his: bool = True):
        """KNN anomaly detector with mahalanobis distance.

        Args:
            init_len (int, optional): The length of initialization data. This can be adjusted by the number of referenced neighbors. Defaults to -1.
            k_neighbor (int, optional): The number of neighbors to cumulate distances. Defaults to 25.
            all_his (bool, optional): The history records used for the reference. True for all records. Defaults to True.
        """

        self.init = []
        self.scores = []
        self.count = 0
        self.k = k_neighbor
        self.buf = []
        self.l = 2 * k_neighbor
        self.use_all = all_his
        if init_len == -1:
            warnings.warn("Set init length to 5 times of k.")
        elif init_len / 2 < k_neighbor:
            warnings.warn("Short init length short, reset it to 4 times of k.")
        self.init_len = max(init_len, 4 * k_neighbor)
        self.window_length = int(self.init_len / 2)
        self.sigma = np.diag(np.ones(self.window_length))

    def _mah_distance(self, x: np.ndarray, item: np.ndarray) -> np.ndarray:
        """Mahalanobis distance

        Args:
            x (np.ndarray): One observation.
            item (np.ndarray): The other observation.

        Returns:
            np.ndarray: Mahalanobis distance
        """
        diff = np.array(x) - np.array(item)
        return np.dot(np.dot(diff, self.sigma), diff.T)

    def _ncm(self, item: np.ndarray, item_in_array: bool = False) -> float:
        """Cumulated Mahalanobis distance among the current observation with all init data.

        Args:
            item (np.ndarray): Current observation.
            item_in_array (bool, optional): Whether the observation in list. Defaults to False.

        Returns:
            float: Cumulated Mahalanobis distance.
        """
        arr = [self._mah_distance(x, item) for x in self.init]
        result = np.sum(
            np.partition(arr, self.k + item_in_array)[: self.k + item_in_array]
        )
        return result

    def fit(self, X: np.ndarray):
        """Record and analyse the current observation from the stream. Detector collect the init data firstly, and further score observation base on the observed data.

        Args:
            X (np.ndarray): Current observation.
        """

        self.count += 1
        self.buf.append(X)

        if self.count < self.window_length:
            pass
        elif self.count < 2 * self.window_length:
            self.init.append(self.buf)
            self.buf = self.buf[1:]
        else:

            ost = self.count % self.init_len
            if ost == 0 or ost == self.window_length:
                try:
                    self.sigma = np.linalg.inv(
                        np.dot(np.array(self.init).T, np.array(self.init))
                    )
                except np.linalg.linalg.LinAlgError:
                    print("\nSingular Matrix at record", self.count)

            if len(self.scores) == 0:
                self.scores = [self._ncm(v, True) for v in self.init]

            new_score = self._ncm(self.buf)

            self.init.pop(0)

            self.init.append(self.buf)

            self.scores.append(new_score)
            self.buf = self.buf[1:]
            if not self.use_all:
                self.scores.pop(0)
        return self

    def score(self, X) -> float:
        """Score the current observation. None for init period and float for the probability of anomalousness.

        Args:
            X (np.ndarray): Current observation.

        Returns:
            float: Anomaly probability.
        """

        if self.count < 2 * self.window_length:
            return None

        score = self.scores[-1]

        prob = 1.0 * len(np.where(np.array(self.scores) < score)[0]) / len(self.scores)
        return prob
