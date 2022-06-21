from streamad.base import BaseDetector
import numpy as np
from math import log
from scipy.optimize import minimize

np.seterr(divide="ignore", invalid="ignore")


class SpotDetector(BaseDetector):
    def __init__(self, window_len: int = 100, prob: float = 1e-4):
        """Univariate Spot model :cite:`DBLP:conf/kdd/SifferFTL17`.

        Args:
            window_len (int, optional): Length of the window for reference. Defaults to 100.
            prob (float, optional): Threshold for probability, a small float value. Defaults to 1e-4.
        """
        super().__init__()

        self.data_type = "univariate"
        self.prob = prob
        self.init_data = []
        self.window_len = window_len
        self._window_len = int(np.sqrt(window_len))
        self.init_length = window_len - self._window_len
        self.num_threshold = {"up": 0, "down": 0}

        nonedict = {"up": None, "down": None}

        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.gamma = dict.copy(nonedict)
        self.sigma = dict.copy(nonedict)

        # self.thup = []
        # self.thdown = []

    def _grimshaw(self, side, epsilon=1e-8, n_points=10):
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
            jac_us = np.divide(1, t, out=np.zeros_like(a), where=t != 0) * (
                1 - vs
            )
            jac_vs = np.divide(1, t, out=np.zeros_like(a), where=t != 0) * (
                -vs + np.mean(1 / s**2)
            )
            return us * jac_vs + vs * jac_us

        Ym = self.peaks[side].min()
        YM = self.peaks[side].max()
        Ymean = self.peaks[side].mean()

        a = np.divide(-1, YM, out=np.array(-1 / epsilon), where=YM != 0)
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * np.divide(
            (Ymean - Ym),
            (Ymean * Ym),
            out=np.array(np.zeros_like(Ymean * Ym) - epsilon),
            where=(Ymean * Ym) != 0,
        )
        c = 2 * np.divide(
            Ymean - Ym,
            Ym**2,
            out=np.array(np.zeros_like(Ym) + epsilon),
            where=Ym != 0,
        )

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
            sigma = np.divide(gamma, z, out=np.zeros_like(gamma), where=z != 0)
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
                g = g + fx**2
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
            L = (
                -n * log(sigma)
                - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
            )
        else:
            L = n * (1 + log(abs(Y.mean()) + 1e-8))

        return L

    def _quantile(self, side, gamma, sigma):

        if side == "up":
            r = (
                (self.init_length - self._window_len)
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
                (self.init_length - self._window_len)
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
        w = sum(self.init_data[: self._window_len])
        M.append(w / self._window_len)
        for i in range(self._window_len, self.index + 1):
            w = w - self.init_data[i - self._window_len] + self.init_data[i]
            M.append(w / self._window_len)

        return np.array(M)

    def _init_drift(self, X: np.ndarray, verbose=False):

        n_init = self.init_length - self._window_len

        M = self._back_mean()

        T = self.init_data[self._window_len :] - M[:-1]
        S = np.sort(T.tolist())
        self.init_threshold["up"] = S[int(0.98 * n_init)]
        self.init_threshold["down"] = S[int(0.02 * n_init)]
        self.peaks["up"] = (
            T[T >= self.init_threshold["up"]] - self.init_threshold["up"]
        )
        self.peaks["down"] = (
            self.init_threshold["down"] - T[T <= self.init_threshold["down"]]
        )

        self.num_threshold["up"] = self.peaks["up"].size
        self.num_threshold["down"] = self.peaks["down"].size

        for side in ["up", "down"]:
            gamma, sigma, _ = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, gamma, sigma)
            self.gamma[side] = gamma
            self.sigma[side] = sigma
        return self

    def _update_one_side(self, side: str, X: float):

        self.peaks[side] = np.append(
            self.peaks[side], abs(X - self.init_threshold[side])
        )
        self.num_threshold[side] += 1
        gamma, sigma, _ = self._grimshaw(side)
        self.extreme_quantile[side] = self._quantile(
            side, gamma=gamma, sigma=sigma
        )

    def fit(self, X: np.ndarray):

        self.init_data.append(float(X))

        if self.index == self.window_len - 1:
            self._init_drift(X)

        if self.index > self.window_len - 1:
            hist_mean = np.mean(self.init_data[-self._window_len :])

            normal_X = float(X) - hist_mean

            if (
                normal_X > self.extreme_quantile["up"]
                or normal_X < self.extreme_quantile["down"]
            ):
                self.init_data = self.init_data[:-1]
            elif normal_X > self.init_threshold["up"]:
                self._update_one_side("up", normal_X)

            elif normal_X < self.init_threshold["down"]:
                self._update_one_side("down", normal_X)

            self.init_data = self.init_data[-self._window_len :]

        return self

    def score(self, X: np.ndarray) -> float:

        hist_mean = np.mean(self.init_data[-self._window_len :])

        normal_X = float(X) - hist_mean

        if (
            normal_X > self.extreme_quantile["up"]
            or normal_X < self.extreme_quantile["down"]
        ):
            score = 1.0

        elif normal_X > self.init_threshold["up"]:
            side = "up"
            score = abs(
                float(self.init_threshold[side] - X)
                / (self.extreme_quantile[side] - self.init_threshold[side])
            )

        elif normal_X < self.init_threshold["down"]:
            side = "down"
            score = abs(
                float(self.init_threshold[side] - X)
                / (self.extreme_quantile[side] - self.init_threshold[side])
            )
        else:
            score = 0.0

        # self.thup.append(self.extreme_quantile["up"] + hist_mean)
        # self.thdown.append(self.extreme_quantile["down"] + hist_mean)

        return float(score)
