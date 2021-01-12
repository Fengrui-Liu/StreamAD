#!/usr/bin/env python
# coding=utf-8

from streamad.base import BaseDetector
import pandas as pd
import numpy as np
from math import log
from scipy.optimize import minimize


class SpotDetector(BaseDetector):
    """Univariate Spot model. :cite:`DBLP:conf/kdd/SifferFTL17`. See `SPOT <https://dl.acm.org/doi/10.1145/3097983.3098144>`_"""

    def __init__(self, prob: float = 1e-4, window_size: int = 10, init_len: int = 300):
        """Spot detector.

        Args:
            prob (float, optional): Threshold for probability, a small float value. Defaults to 1e-4.
            window_size (int, optional): A window for reference. Defaults to 10.
            init_len (int, optional): Data length for initialization. Recommended > 100. Defaults to 300.
        """
        self.prob = prob
        self.data = []
        self.init_data = []
        self.init_length = init_len
        self.record_count = 0
        self.num_threshold = {"up": 0, "down": 0}
        self.window_size = window_size

        nonedict = {"up": None, "down": None}

        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.gamma = dict.copy(nonedict)
        self.sigma = dict.copy(nonedict)

        self.thup = []
        self.thdown = []

    def _grimshaw(self, side, epsilon=1e-8, n_points=8):
        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks[side].min()
        YM = self.peaks[side].max()
        Ymean = self.peaks[side].mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        left_zeros = self._rootsFinder(
            lambda t: w(self.peaks[side], t),
            lambda t: jac_w(self.peaks[side], t),
            (a + epsilon, -epsilon),
            n_points,
            "regular",
        )

        right_zeros = self._rootsFinder(
            lambda t: w(self.peaks[side], t),
            lambda t: jac_w(self.peaks[side], t),
            (b, c),
            n_points,
            "regular",
        )

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = self._log_likelihood(self.peaks[side], gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks[side]) - 1
            sigma = gamma / z
            ll = self._log_likelihood(self.peaks[side], gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _rootsFinder(self, fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
                    scalar function
        jac : function
            first order derivative of the function
        bounds : tuple
            (min,max) interval for the roots search
        npoints : int
            maximum number of roots to output
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            try:
                X0 = np.arange(bounds[0] + step, bounds[1], step)
            except:
                X0 = np.random.uniform(bounds[0], bounds[1], npoints)
        elif method == "random":
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(
            lambda X: objFun(X, fun, jac),
            X0,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(X0),
        )

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(self, Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
                    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _quantile(self, side, gamma, sigma):

        if side == "up":
            r = (
                (self.init_length - self.window_size)
                * self.prob
                / self.num_threshold[side]
            )
            if gamma != 0:
                return self.init_threshold["up"] + (sigma / gamma) * (
                    pow(r, -gamma) - 1
                )
            else:
                return self.init_threshold["up"] - sigma * log(r)
        elif side == "down":
            r = (
                (self.init_length - self.window_size)
                * self.prob
                / self.num_threshold[side]
            )
            if gamma != 0:
                return self.init_threshold["down"] - (sigma / gamma) * (
                    pow(r, -gamma) - 1
                )
            else:
                return self.init_threshold["down"] + sigma * log(r)
        else:
            raise ValueError("The side is not right")

    def _back_mean(self):
        M = []
        w = sum(self.init_data[: self.window_size])
        M.append(w / self.window_size)
        for i in range(self.window_size, self.record_count):
            w = w - self.init_data[i - self.window_size] + self.init_data[i]
            M.append(w / self.window_size)

        return np.array(M)

    def _init_drift(self, X: pd.Series, verbose=False):

        n_init = self.init_length - self.window_size

        M = self._back_mean()

        T = self.init_data[self.window_size :] - M[:-1]
        S = np.sort(T.tolist())
        self.init_threshold["up"] = S[int(0.98 * n_init)]
        self.init_threshold["down"] = S[int(0.02 * n_init)]
        self.peaks["up"] = T[T > self.init_threshold["up"]] - self.init_threshold["up"]
        self.peaks["down"] = (
            self.init_threshold["down"] - T[T < self.init_threshold["down"]]
        )

        self.num_threshold["up"] = self.peaks["up"].size
        self.num_threshold["down"] = self.peaks["down"].size

        for side in ["up", "down"]:
            gamma, sigma, _ = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, gamma, sigma)
            self.gamma[side] = gamma
            self.sigma[side] = sigma
        return self

    def fit(self, X: np.ndarray):
        """Record and analyse the current observation from the stream. Detector collect the init data firstly, and add random drift.

        Args:
            X (np.ndarray): [description]

        """

        self.record_count += 1
        self.init_data.append(float(X))

        if self.record_count == self.init_length:
            self._init_drift(X)

        return self

    def score(self, X: np.ndarray) -> float:
        """Score the current observation. None for init period and float for the probability of anomalousness between two thresholds. 1.0 for certain anomaly point.

        Args:
            X (np.ndarray): Current observation.

        Returns:
            float: Anomaly probability.
        """
        if self.record_count <= self.init_length:
            return None

        hist_mean = np.mean(self.init_data[-self.window_size :])

        normal_X = float(X) - hist_mean

        prob = 0.0

        if normal_X > self.extreme_quantile["up"]:
            prob = 1.0
            self.init_data = self.init_data[:-1]
        elif normal_X > self.init_threshold["up"]:
            prob = float(normal_X - self.init_threshold["up"]) / (
                self.extreme_quantile["up"] - self.init_threshold["up"]
            )
            self.peaks["up"] = np.append(
                self.peaks["up"], normal_X - self.init_threshold["up"]
            )
            self.num_threshold["up"] += 1
            gamma, sigma, _ = self._grimshaw("up")
            self.extreme_quantile["up"] = self._quantile("up", gamma=gamma, sigma=sigma)

        elif normal_X < self.extreme_quantile["down"]:
            prob = 1.0
            self.init_data = self.init_data[:-1]

        elif normal_X < self.init_threshold["down"]:
            prob = float(self.init_threshold["down"] - normal_X) / (
                self.extreme_quantile["down"] - normal_X
            )
            self.peaks["down"] = np.append(
                self.peaks["down"], self.init_threshold["down"] - normal_X
            )

            self.num_threshold["down"] += 1

            gamma, sigma, _ = self._grimshaw("down")
            self.extreme_quantile["down"] = self._quantile(
                "down", gamma=gamma, sigma=sigma
            )
        else:
            prob = 0.0

        self.init_data = self.init_data[-self.window_size :]

        self.thup.append(self.extreme_quantile["up"] + hist_mean)
        self.thdown.append(self.extreme_quantile["down"] + hist_mean)

        return prob
