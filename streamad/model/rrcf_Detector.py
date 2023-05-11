from collections import deque

import numpy as np
import rrcf
from streamad.base import BaseDetector
from copy import deepcopy


class RrcfDetector(BaseDetector):
    def __init__(self, num_trees=10, tree_size=12, **kwargs):
        """Rrcf detector :cite:`DBLP:conf/icml/GuhaMRS16`.

        Args:
            window_len (int, optional): Length of sliding window. Defaults to 50.
            num_trees (int, optional): Number of trees. Defaults to 10.
            tree_size (int, optional): Size of each tree. Defaults to 12.
        """

        super().__init__(data_type="multivariate", **kwargs)
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            self.forest.append(tree)
        self.avg_codisp = {}

        self.shingle = deque(maxlen=int(np.sqrt(self.window_len)))

    def fit(self, X: np.ndarray, timestamp: int = None):
        self.shingle.append(X)

        if not self.forest[0].ndim:
            dim = X.shape[0]
            for tree in self.forest:
                tree.ndim = dim

        if self.shingle.maxlen == len(self.shingle):
            if self.index > (self.shingle.maxlen + self.tree_size):
                list(
                    map(
                        lambda x: x.forget_point(self.index - self.tree_size),
                        self.forest,
                    )
                )

            list(
                map(
                    lambda x: x.insert_point(self.shingle, self.index),
                    self.forest,
                )
            )

        return self

    def score(self, X: np.ndarray, timestamp: int = None):
        score_list = list(map(lambda x: x.codisp(self.index), self.forest))

        score = sum(score_list) / self.num_trees

        return float(score)


if __name__ == "__main__":
    import cProfile
    from line_profiler import LineProfiler

    lp = LineProfiler()

    model = RrcfDetector()

    # lp.add_function(_Chain.fit)
    # lp.add_function(_Chain.score)
    # lp.add_function(_Chain.bincount)
    lp.add_function(model.fit)
    lp.add_function(model.score)
    lp_wrapper = lp(model.fit_score)

    for i in range(1500):
        lp_wrapper(np.array([i]))
        # model.fit_score(np.array([i]))

    lp.print_stats()
