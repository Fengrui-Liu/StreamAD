from collections import deque

import numpy as np
import rrcf
from streamad.base import BaseDetector


class RrcfDetector(BaseDetector):
    def __init__(self, window_len=10, num_trees=40, tree_size=256):
        """Rrcf detector :cite:`DBLP:conf/icml/GuhaMRS16`.

        Args:
            window_len (int, optional): Length of sliding window. Defaults to 10.
            num_trees (int, optional): Number of trees. Defaults to 40.
            tree_size (int, optional): Size of each tree. Defaults to 256.
        """

        super().__init__()
        self.num_trees = num_trees
        self.window_len = window_len
        self.tree_size = tree_size
        self.forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            self.forest.append(tree)
        self.avg_codisp = {}

        self.shingle = deque(maxlen=window_len)
        self.score_list = []

    def fit(self, X: np.ndarray):

        self.shingle.append(X[0])

        if self.index < self.window_len:
            return self

        self.score_list = []

        for tree in self.forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self.index - self.tree_size)

            tree.insert_point(self.shingle, self.index)

            self.score_list.append(tree.codisp(self.index))

        return self

    def score(self, X: np.ndarray) -> float:

        if self.index < self.window_len:
            return None

        score = sum(self.score_list) / len(self.score_list)

        return score
