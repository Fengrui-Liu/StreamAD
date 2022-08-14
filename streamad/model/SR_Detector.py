import numpy as np
from streamad.base import BaseDetector
from collections import deque
from copy import deepcopy

EPS = 1e-8


class SRDetector(BaseDetector):
    def __init__(
        self,
        window_len: int = 100,
        extend_len: int = 5,
        ahead_len: int = 10,
        mag_num: int = 5,
    ):
        """Spectral Residual Detector :cite:`DBLP:conf/kdd/RenXWYHKXYTZ19`.

        Args:
            window_len (int, optional): Length of sliding window. Defaults to 100.
            extend_len (int, optional): Length to be extended, for FFT transforme. Defaults to 5.
            ahead_len (int, optional): Length to look ahead for references. Defaults to 10.
            mag_num (int, optional): Number of FFT magnitude. Defaults to 5.
        """
        super().__init__()
        self.window = deque(maxlen=window_len)
        self.window_len = window_len
        self.extend_len = extend_len
        assert ahead_len > 1, "ahead_len must be greater than 1"
        self.ahead_len = ahead_len
        self.mag_num = mag_num

    def fit(self, X):
        self.window.append(X[0])

        return self

    def score(self, X):

        window = deepcopy(self.window)

        window.pop()
        window.append(X[0])

        extended_window = self._extend_window(window)

        mags = self._sr_transform(extended_window)
        anomaly_scores = self._spectral_score(mags)

        return anomaly_scores[-1 - self.extend_len]

    def _spectral_score(self, mags):

        avg_mag = self._average_filter(mags, n=self.mag_num * 10)
        safeDivisors = np.clip(avg_mag, EPS, avg_mag.max())

        raw_scores = np.divide(
            np.abs(mags - avg_mag),
            safeDivisors,
            out=np.zeros_like(mags),
            where=safeDivisors != 0,
        )
        scores = np.clip(raw_scores / 10.0, 0, 1.0)

        return scores

    def _sr_transform(self, window):

        trans = np.fft.fft(window)
        mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
        eps_index = np.where(mag <= EPS)[0]
        mag[eps_index] = EPS

        mag_log = np.log(mag)
        mag_log[eps_index] = 0

        spectral = np.exp(
            mag_log - self._average_filter(mag_log, n=self.mag_num)
        )

        trans.real = trans.real * spectral / mag
        trans.imag = trans.imag * spectral / mag

        trans.real[eps_index] = 0
        trans.imag[eps_index] = 0

        wave_r = np.fft.ifft(trans)

        mag = np.sqrt(wave_r.real ** 2, wave_r.imag ** 2)

        return mag

    def _average_filter(self, values, n=3):
        if n >= len(values):
            n = len(values)

        res = np.cumsum(values, dtype=float)
        res[n:] = res[n:] - res[:-n]
        res[n:] = res[n:] / n

        for i in range(1, n):
            res[i] /= i + 1

        return res

    def _extend_window(self, window):

        predicted_window = [
            self._predict_next(list(window)[-self.ahead_len : -1])
        ] * self.extend_len

        extended_window = np.concatenate((window, predicted_window), axis=0)

        return extended_window

    def _predict_next(self, ahead_window):

        assert (
            len(ahead_window) > 1
        ), "ahead window must have at least 2 elements"

        ele_last = ahead_window[-1]
        n = len(ahead_window)

        slopes = [
            (ele_last - ele) / (n - 1 - i)
            for i, ele in enumerate(ahead_window[:-1])
        ]

        return ahead_window[1] + sum(slopes)
