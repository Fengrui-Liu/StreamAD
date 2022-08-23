import numpy as np
from streamad.base import BaseDetector
from streamad.util import StreamStatistic


class Leaf:
    def __init__(
        self, left=None, right=None, depth=0,
    ):
        self.left = left
        self.right = right
        self.r = 0
        self.l = 0
        self.split_attrib = 0
        self.split_value = 0.0
        self.k = depth


class HSTreeDetector(BaseDetector):
    def __init__(self, tree_height: int = 10, tree_num: int = 100, **kwargs):
        """Half space tree detectors. :cite:`DBLP:conf/ijcai/TanTL11`.

        Args:
            tree_height (int, optional): Height of a half space tree. Defaults to 10.
            tree_num (int, optional): Totla number of the trees. Defaults to 100.
        """
        super().__init__(data_type="multivariate", **kwargs)
        self.tree_height = tree_height
        self.tree_num = tree_num
        self.forest = []
        self.data_stats = StreamStatistic()

        self.dimensions = None

    def _generate_max_min(self):
        max_arr = np.zeros(self.dimensions)
        min_arr = np.zeros(self.dimensions)
        for q in range(self.dimensions):
            s_q = np.random.random_sample()
            max_value = max(s_q, 1 - s_q)
            max_arr[q] = s_q + max_value
            min_arr[q] = s_q - max_value

        return max_arr, min_arr

    def _init_a_tree(self, max_arr, min_arr, k):
        if k == self.tree_height:
            return Leaf(depth=k)

        leaf = Leaf()
        q = np.random.randint(self.dimensions)
        p = (max_arr[q] + min_arr[q]) / 2.0
        temp = max_arr[q]
        max_arr[q] = p
        leaf.left = self._init_a_tree(max_arr, min_arr, k + 1)
        max_arr[q] = temp
        min_arr[q] = p
        leaf.right = self._init_a_tree(max_arr, min_arr, k + 1)
        leaf.split_attrib = q
        leaf.split_value = p
        leaf.k = k
        return leaf

    def _update_tree_mass(self, tree, X, is_ref_window):
        if tree:
            if tree.k != 0:
                if is_ref_window:
                    tree.r += 1

                tree.l += 1
            if X[tree.split_attrib] > tree.split_value:
                tree_new = tree.right
            else:
                tree_new = tree.left
            self._update_tree_mass(tree_new, X, is_ref_window)

    def _reset_tree(self, tree):
        if tree:
            tree.r = tree.l
            tree.l = 0
            self._reset_tree(tree.left)
            self._reset_tree(tree.right)

    def fit(self, X: np.ndarray) -> None:

        self.data_stats.update(X)

        X_normalized = np.divide(
            X - self.data_stats.get_min(),
            self.data_stats.get_max() - self.data_stats.get_min(),
            out=np.zeros_like(X),
            where=self.data_stats.get_max() - self.data_stats.get_min() != 0,
        )
        X_normalized[np.abs(X_normalized) == np.inf] = 0

        if self.dimensions is None:
            self.dimensions = len(X)
            for _ in range(self.tree_num):
                max_arr, min_arr = self._generate_max_min()
                tree = self._init_a_tree(max_arr, min_arr, 0)
                self.forest.append(tree)

        if self.index < self.window_len:
            for tree in self.forest:
                self._update_tree_mass(tree, X_normalized, True)
        else:
            if self.index % self.window_len == 0:
                for tree in self.forest:
                    self._reset_tree(tree)

            for tree in self.forest:
                self._update_tree_mass(tree, X_normalized, False)

        return self

    def score(self, X: np.ndarray) -> float:

        score = 0.0

        X_normalized = np.divide(
            X - self.data_stats.get_min(),
            self.data_stats.get_max() - self.data_stats.get_min(),
            out=np.zeros_like(X),
            where=self.data_stats.get_max() - self.data_stats.get_min() != 0,
        )
        X_normalized[np.abs(X_normalized) == np.inf] = 0

        for tree in self.forest:
            score += self._score_tree(tree, X_normalized, 0)

        score = score / self.tree_num

        return float(score)

    def _score_tree(self, tree, X, k):
        s = 0
        if not tree:
            return s

        s += tree.r * (2 ** k)

        if X[tree.split_attrib] > tree.split_value:
            tree_new = tree.right
        else:
            tree_new = tree.left

        s += self._score_tree(tree_new, X, k + 1)

        return s
