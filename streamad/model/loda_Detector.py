from collections import deque

import numpy as np
from streamad.base import BaseDetector
from fast_histogram import histogram1d


class LodaDetector(BaseDetector):
    def __init__(self, random_cuts_num: int = 10, **kwargs):
        """Multivariate LODA Detector :cite:`DBLP:journals/ml/Pevny16`.

        Args:
            window_len (int, optional): The length of window. Defaults to 50.
            random_cuts_num (int, optional): The number of random experiments. Defaults to 10.
        """
        super().__init__(data_type="multivariate", **kwargs)

        self.random_cuts_num = random_cuts_num
        self.bins_num = int(
            1 * (self.window_len**1) * (np.log(self.window_len) ** -1)
        )
        self._weights = np.ones(random_cuts_num) / random_cuts_num
        self.components_num = None
        self.nonzero_components_num = None
        self.zero_components_num = None
        self._projections = None
        self._histograms = None
        self._limits = None

    def fit(self, X: np.ndarray, timestamp: int = None):
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

                try:
                    self._histograms[i, :] = (
                        histogram1d(
                            projected_data,
                            range=(
                                projected_data.min(),
                                projected_data.max() + 1e-12,
                            ),
                            bins=self.bins_num,
                        )
                        + 1e-12
                    )
                except:
                    self._histograms[i, :] = (
                        histogram1d(
                            projected_data,
                            range=(
                                projected_data.min(),
                                projected_data.max() + 1e-5,
                            ),
                            bins=self.bins_num,
                        )
                        + 1e-12
                    )
                self._limits[i, :] = np.linspace(
                    projected_data.min(),
                    projected_data.max() + 1e-12,
                    num=self.bins_num + 1,
                )

                self._histograms[i, :] /= np.sum(self._histograms[i, :])

        return self

    def score(self, X: np.ndarray, timestamp: int = None):
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


if __name__ == "__main__":
    import cProfile
    import resource

    # from line_profiler import LineProfiler

    # lp = LineProfiler()

    model = LodaDetector()

    # lp.add_function(model.fit)
    # lp.add_function(model.score)
    # lp_wrapper = lp(model.fit_score)
    import sys

    for i in range(1500):
        # lp_wrapper(np.array([i]))
        model.fit_score(np.array([i * 10]))

        r = sys.getsizeof(model)
        # r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        print(r)

    # lp.print_stats()
