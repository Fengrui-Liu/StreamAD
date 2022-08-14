import warnings
import itertools
import numpy as np
warnings.filterwarnings("ignore")
import pandas as pd
import statsmodels.api as sm
from collections import deque

from streamad.base.detector import BaseDetector


class ArimaDetector(BaseDetector):
    def __init__(
        self,
        window_length: int = 200,
    ):
        """Auto Regressive Integrated Moving Averages Detector :cite: ``

        Args:
            window_length (int, optional): Length of sliding window. Defaults to 200.
        """
        super().__init__()
        self.window_len = window_length
        self.buf = deque(maxlen=self.window_len + 1)
        self.best_result = None
        self.count = 0
        self.best_model = None
        self.best_order = None
        self.best_seasonal_order = None


    def fit(self, X: np.ndarray) -> None:
        warnings.filterwarnings("ignore")
        self.count = self.count + 1
        if self.count > self.window_len :
            X_train_c = np.array(self.buf)
            X_train_c = X_train_c.reshape(X_train_c.shape[0],1)
            X_train = pd.DataFrame(X_train_c)
            self.buf.append(X)
            self.buf.popleft()
            if self.count == self.window_len + 1 :
                results = []
                best_aic = float("inf")
                p = d = q = range(0, 2)
                pdq = list(itertools.product(p, d, q))
                seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
                for param in pdq:
                    for param_seasonal in seasonal_pdq:
                        model = sm.tsa.statespace.SARIMAX(X_train.astype(float), order=param, seasonal_order=param_seasonal,enforce_stationarity=False, enforce_invertibility=False)
                        result = model.fit(disp=0)
                        aic = result.aic
                        if aic < best_aic:
                            best_model = model
                            best_aic = aic
                            self.best_order = param
                            self.best_seasonal_order = param_seasonal
                        results.append([self.best_order, self.best_seasonal_order, result.aic])
                self.best_result = best_model.fit(disp=0)
            else:
                self.best_model = sm.tsa.statespace.SARIMAX(X_train.astype(float), order=self.best_order, seasonal_order=self.best_seasonal_order,enforce_stationarity=False, enforce_invertibility=False)
                self.best_result = self.best_model.fit(disp=0)
        else:
            self.buf.append(X)
        return self


    def score(self, X: np.ndarray) -> float:
        pred_uc = self.best_result.get_forecast(steps=1)
        pred_ci = pred_uc.conf_int()
        pred_mid = (pred_ci.iloc[0, 0]+pred_ci.iloc[0, 1])/2
        pred_range = pred_ci.iloc[0, 1]-pred_ci.iloc[0, 0]
        if self.count <= self.window_len:
            return None
        elif pred_ci.iloc[0, 0] > X :
            score = abs((X-pred_mid)/pred_range)
            return score
        elif X > pred_ci.iloc[0, 1] :
            score = abs((X-pred_mid)/pred_range)
            return score
        else:
            score = 0
            return float(score)
               