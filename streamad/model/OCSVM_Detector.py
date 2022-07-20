from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from streamad.base.detector import BaseDetector


class OCSVMDetector(BaseDetector):
    def __init__(
        self,
        nu: float=0.5,
        kernel="rbf",
        window_length: int=10,
    ):
        """One-Class SVM Detector :cite:`conf/icde/KhelifatiKCHLH21`.

        Args:
            nu (float, optional): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Defaults to 0.5.
            kernel (str, optional): The kernel type to be used in the algorithm. Defaults to "rbf".
            window_length (int, optional): Length of sliding window. Defaults to 10.
        """
        super().__init__()
        self.nu = nu
        self.kernel = kernel
        self.buf = []
        self.count = 0
        self.window_length = window_length
        self.model = None

    def fit(self, X: np.ndarray):

        self.count += 1
        X_train = self.buf
        self.buf.append(X)
        if self.count <= self.window_length:
            pass
        else:
            self.buf = self.buf[1:]
            scaler = StandardScaler()
            np_scaled = scaler.fit_transform(X_train)
            data = pd.DataFrame(np_scaled)
            # train oneclassSVM
            self.model = OneClassSVM(gamma="scale", nu=self.nu, kernel=self.kernel)
            self.model.fit(data)

        return self

    def score(self, X: np.ndarray) -> float:

        if self.count <= self.window_length:
            return None
        else:
            score = self.model.decision_function(X.reshape(1, -1))
        return abs(score)
