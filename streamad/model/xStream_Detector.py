from streamad.base import BaseDetector
import pandas as pd
import numpy as np
import mmh3


class xStreamDetector(BaseDetector):
    """Multivariate xStreamDetector :cite:`DBLP:conf/kdd/ManzoorLA18`. See `xStream <https://cmuxstream.github.io/>`_"""

    def __init__(
        self,
        n_components: int = 50,
        n_chains: int = 100,
        depth: int = 25,
        window_size: int = 25,
    ):
        """xStream detector for multivariate data.

        Args:
            n_components (int, optional): Number of streamhash projection, similar to feature numbers. Defaults to 50.
            n_chains (int, optional): Number of half-space chains. Defaults to 100.
            depth (int, optional): Maximum depth for each chain. Defaults to 25.
            window_size (int, optional): Size of reference window. Defaults to 25.
        """

        self.projector = StreamhashProjector(
            num_components=n_components, density=1 / 3.0
        )
        self.window_size = window_size
        self.count = 0
        self.cur_window = []
        self.ref_window = []
        delta = np.ones(n_components) * 0.5
        self.hs_chains = _hsChains(deltamax=delta, n_chains=n_chains, depth=depth)
        self.scores = []

    def fit(self, X: np.ndarray):
        """xStreamDetector collects data with a window length, projects them via streamhash projector, and then scores the observed data.

        Args:
            X (np.ndarray): Current observation.
        """
        self.count += 1

        projected_X = self.projector.transform(X)
        self.cur_window.append(projected_X)
        self.hs_chains.fit(projected_X)

        if self.count % self.window_size == 0:
            self.ref_window = self.cur_window
            self.cur_window = []

            deltamax = np.ptp(self.ref_window, axis=0) / 2.0
            deltamax[np.abs(deltamax) <= 0.0001] = 1.0

            self.hs_chains.set_deltamax(deltamax=deltamax)
            self.hs_chains.next_window()

        return self

    def score(self, X: np.ndarray) -> float:
        """Score the current observation. None for init period and float for the probability of anomalousness.

        Args:
            X (np.ndarray): Current observation.

        Returns:
            float: Anomaly probability.
        """
        projected_X = self.projector.transform(X)

        score = -1.0 * self.hs_chains.score_chains(projected_X)
        self.scores.append(score)

        if self.count < self.window_size:
            return None

        score = self.scores[-1]

        prob = 1.0 * len(np.where(np.array(self.scores) < score)[0]) / len(self.scores)

        return prob


class _Chain:
    def __init__(self, deltamax, depth):

        self.depth = depth
        self.deltamax = deltamax
        self.rand = np.random.rand(len(deltamax))
        self.rand_shift = self.rand * deltamax
        self.cmsketch_ref = [{} for _ in range(depth)] * depth
        self.cmsketch_cur = [{} for _ in range(depth)] * depth
        self.is_first_window = True
        self.fs = [np.random.randint(0, len(deltamax)) for _ in range(depth)]

    def bincount(self, X):

        scores = np.zeros(self.depth)
        prebins = np.zeros(X.shape[0], dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1
            if depthcount[f] == 1:
                prebins[f] = X[f] + self.rand_shift[f] / self.deltamax[f]
            else:
                prebins[f] = 2.0 * prebins[f] - self.rand_shift[f] / self.deltamax[f]

            cmsketch = self.cmsketch_ref[depth]

            for prebin in prebins:

                l = int(prebin)
                if l in cmsketch:
                    scores[depth] = cmsketch[l]
                else:
                    scores[depth] = 0.0

        return scores

    def score(self, X):

        scores = self.bincount(X)

        depths = np.arange(1, self.depth + 1)

        scores = np.log2(1.0 + scores) + depths
        return np.min(scores)

    def fit(self, X):

        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[f] = (X[f] + self.rand_shift[f]) / self.deltamax[f]
            else:
                prebins[f] = 2.0 * prebins[f] - self.rand_shift[f] / self.deltamax[f]

            if self.is_first_window:

                cmsketch = self.cmsketch_ref[depth]
                for prebin in prebins:

                    l = int(prebin)

                    if l not in cmsketch:
                        cmsketch[l] = 0
                    cmsketch[l] += 1

                self.cmsketch_ref[depth] = cmsketch
                self.cmsketch_cur[depth] = cmsketch
            else:
                cmsketch = self.cmsketch_cur[depth]
                for prebin in prebins:
                    l = int(prebin)

                    if l not in cmsketch:
                        cmsketch[l] = 0
                    cmsketch[l] += 1
                self.cmsketch_cur[depth] = cmsketch

        return self

    def next_window(self):
        self.is_first_window = False
        self.cmsketch_ref = self.cmsketch_cur
        self.cmsketch_cur = [{} for _ in range(self.depth)] * self.depth


class _hsChains:
    def __init__(self, deltamax, n_chains: int = 100, depth: int = 25) -> None:
        self.nchains = n_chains
        self.depth = depth
        self.chains = [_Chain(deltamax, depth) for _ in range(n_chains)]

    def score_chains(self, X):

        scores = 0
        for chain in self.chains:
            scores += chain.score(X)

        scores = float(scores) / float(self.nchains)

        return scores

    def fit(self, X):
        for chain in self.chains:
            chain.fit(X)

    def next_window(self):
        for chain in self.chains:
            chain.next_window()

    def set_deltamax(self, deltamax):
        for chain in self.chains:
            chain.deltamax = deltamax
            chain.rand_shift = chain.rand * deltamax


class StreamhashProjector:
    def __init__(self, num_components, density=1 / 3.0):

        self.keys = np.arange(0, num_components, 1)
        self.constant = np.sqrt(1.0 / density) / np.sqrt(num_components)
        self.density = density
        self.n_components = num_components

    def transform(self, X):
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

        hash_value = int(mmh3.hash(s, signed=False, seed=k)) / (2.0 ** 32 - 1)
        s = self.density
        if hash_value <= s / 2.0:
            return -1 * self.constant
        elif hash_value <= s:
            return self.constant
        else:
            return 0
