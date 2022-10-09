import itertools
import warnings

import numpy as np
import statsmodels.api as sm
from streamad.base.detector import BaseDetector

warnings.filterwarnings("ignore")


class SArimaDetector(BaseDetector):
    def __init__(self, **kwargs):
        """Auto Regressive Integrated Moving Averages Detector :cite:`durbin2012time`

        Args:
            window_len (int, optional): Length of sliding window. Defaults to 200.
        """
        super().__init__(data_type="univariate", **kwargs)
        self.best_result = None
        self.best_model = None
        self.best_order = None
        self.best_seasonal_order = None

    def _init_fit(self):

        best_aic = float("inf")
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [
            (x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))
        ]
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                model = sm.tsa.statespace.SARIMAX(
                    list(self.window),
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=0)
                aic = result.aic
                if aic < best_aic:
                    self.best_model = model
                    best_aic = aic
                    self.best_order = param
                    self.best_seasonal_order = param_seasonal

        self.best_result = self.best_model.fit(disp=0)

    def fit(self, X: np.ndarray, timestamp: int = None):

        self.window.append(X[0])
        if self.index == self.window_len:
            self._init_fit()

        if self.index > self.window_len:
            self.best_result = self.best_result.append(X)

        return self

    def score(self, X: np.ndarray, timestamp: int = None):

        pred_uc = self.best_result.get_forecast(steps=1)

        pred_ci = pred_uc.conf_int()
        pred_mid = (pred_ci[0, 0] + pred_ci[0, 1]) / 2
        pred_range = pred_ci[0, 1] - pred_ci[0, 0]

        if pred_ci[0, 0] > X:
            score = abs((X - pred_mid) / pred_range)
            return score
        elif X > pred_ci[0, 1]:
            score = abs((X - pred_mid) / pred_range)
            return score
        else:
            score = 0
            return float(score)
