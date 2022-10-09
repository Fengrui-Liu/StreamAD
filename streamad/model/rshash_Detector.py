import numpy as np
from streamad.base import BaseDetector
from streamad.util import StreamStatistic
from collections import deque


class RShashDetector(BaseDetector):
    def __init__(
        self, decay=0.015, components_num=100, hash_num: int = 10, **kwargs
    ):
        """Multivariate RSHashDetector :cite:`DBLP:conf/icdm/SatheA16`.

        Args:
            window_len (int, optional): Length of data to burn in/init. Defaults to 150.
            decay (float, optional): Decay ratio. Defaults to 0.015.
            components_num (int, optional): Number of components. Defaults to 100.
            hash_num (int, optional): Number of hash functions. Defaults to 10.
        """
        super().__init__(data_type="multivariate", **kwargs)

        self.decay = decay
        self.data_stats = StreamStatistic()

        self.hash_num = hash_num
        self.components_num = components_num
        self.cmsketches = [{} for _ in range(hash_num)]

        self.alpha = None

        self.effective_s = max(1000, 1.0 / (1 - np.power(2, -self.decay)))
        self.f = np.random.uniform(
            low=1.0 / np.sqrt(self.effective_s),
            high=1 - (1.0 / np.sqrt(self.effective_s)),
            size=self.components_num,
        )

    def _burn_in(self):

        # Normalized the init data
        buffer = np.array(self.window)
        buffer_normalized = np.divide(
            buffer - self.data_stats.get_min(),
            self.data_stats.get_max() - self.data_stats.get_min(),
            out=np.zeros_like(buffer),
            where=self.data_stats.get_max() - self.data_stats.get_min() != 0,
        )
        buffer_normalized[np.abs(buffer_normalized) == np.inf] = 0

        for r in range(self.components_num):
            for i in range(buffer.shape[0]):
                Y = np.floor(
                    (buffer_normalized[i, :] + np.array(self.alpha[r]))
                    / self.f[r]
                )

                mod_entry = np.insert(Y, 0, r)
                mod_entry = tuple(mod_entry.astype(int))

                for w in range(self.hash_num):
                    try:
                        value = self.cmsketches[w][mod_entry]
                    except KeyError:
                        value = (0, 0)

                    value = (0, value[1] + 1)
                    self.cmsketches[w][mod_entry] = value

    def fit(self, X: np.ndarray, timestamp: int = None):

        if self.index == 0:
            self.alpha = [
                np.random.uniform(low=0, high=self.f[r], size=len(X))
                for r in range(self.components_num)
            ]

        self.data_stats.update(X)

        if self.index == self.window.maxlen - 1:
            self._burn_in()

        if len(self.window) < self.window.maxlen:
            self.window.append(X)
            return self

        return self

    def score(self, X: np.ndarray, timestamp: int = None) -> float:

        X_normalized = np.divide(
            X - self.data_stats.get_min(),
            self.data_stats.get_max() - self.data_stats.get_min(),
            out=np.zeros_like(X),
            where=self.data_stats.get_max() - self.data_stats.get_min() != 0,
        )
        X_normalized[np.abs(X_normalized) == np.inf] = 0

        score_instance = 0

        for r in range(self.components_num):
            Y = np.floor((X_normalized + np.array(self.alpha[r])) / self.f[r])
            mod_entry = np.insert(Y, 0, r)
            mod_entry = tuple(mod_entry.astype(int))

            c = []

            for w in range(len(self.cmsketches)):
                try:
                    value = self.cmsketches[w][mod_entry]
                except KeyError:
                    value = (self.index, 0)

                tstamp = value[0]
                wt = value[1]
                new_wt = wt * np.power(2, -self.decay * (self.index - tstamp))
                c.append(new_wt)
                new_tstamp = self.index
                self.cmsketches[w][mod_entry] = (new_tstamp, new_wt + 1)

            min_c = min(c)
            c = np.log(1 + min_c)
            score_instance += c

        score = score_instance / self.components_num

        return float(score)
