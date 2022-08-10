import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelmax


class ThetaMultiPeriodicity:
    def __init__(self, pval: float = 0.05, max_lag: int = None) -> None:
        """Theta-method for extract multi-periodicity. :cite:`assimakopoulos2000theta`

        Args:
            pval (float, optional): Signifiance level. Defaults to 0.05.
            max_lag (int, optional): The max number of lag. Defaults to None.
        """

        self.pval = 0.05
        self.max_lag = max_lag

    def get_multi_periodicity(self, X: np.ndarray) -> None:
        """
        Calculate the multi-periodicity of the data.
        """
        tcrit = norm.ppf(1 - self.pval / 2)

        if self.max_lag is None:
            self.max_lag = max(
                min(int(10 * np.log10(X.shape[0])), X.shape[0] - 1), 100
            )

        xacf = acf(X, nlags=self.max_lag, fft=True)
        xacf[np.isnan(xacf)] = 0

        candidates = np.intersect1d(np.where(xacf > 0), argrelmax(xacf)[0])

        # candidates = candidates[candidates < int(X.shape[0] / 3)]

        if candidates.shape[0] == 0:
            return []

        else:
            candidates = candidates[
                np.insert(argrelmax(xacf[candidates])[0], 0, 0)
            ]

        xacf = xacf[1:]
        clim = (
            tcrit
            / np.sqrt(X.shape[0])
            * np.sqrt(np.cumsum(np.insert(np.square(xacf) * 2, 0, 1)))
        )

        candidate_filter = candidates[
            xacf[candidates - 1] > clim[candidates - 1]
        ]

        candidate_filter = sorted(
            candidate_filter.tolist(), key=lambda c: xacf[c - 1], reverse=True
        )

        return candidate_filter
