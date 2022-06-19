from streamad.base import BaseMetrics
from streamad.evaluate.ts_metrics import TSMetric
import numpy as np


class SeriesAwareMetircs(BaseMetrics):
    def __init__(
        self,
        anomaly_threshold: float = 0.8,
        beta: float = 1.0,
        bias_p: str = "flat",
        bias_r: str = "flat",
    ):
        """Time series aware metrics :cite:`DBLP:conf/nips/TatbulLZAG18`

        Args:
            anomaly_threshold (float, optional): A threshold to determine the anomalies, it can covert the anomaly scores to binary (0/1) indicators. Defaults to 0.8.
            beta (float, optional):  F-beta score, like a F1-score. Defaults to 1.0.
            bias_p (str, optional): Bias for precision. Optionals are "flat", "front", "middle", "back". Defaults to "flat".
            bias_r (str, optional): Bias for recall. Optionals are "flat", "front", "middle", "back". Defaults to "flat".
        """
        super().__init__()
        self.threshold = anomaly_threshold
        self.beta = beta
        self.bias_p = bias_p
        self.bias_r = bias_r
        self.precision = None
        self.recall = None
        self.Fbeta = None

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        super().evaluate(y_true, y_pred)

        select = self.y_pred > self.threshold
        self.y_pred[select] = 1
        self.y_pred[~select] = 0

        metric = TSMetric(
            metric_option="time-series",
            beta=self.beta,
            alpha_r=0.0,
            cardinality="reciprocal",
            bias_p=self.bias_p,
            bias_r=self.bias_r,
        )
        self.precision, self.recall, self.Fbeta = metric.score(
            self.y_true, self.y_pred
        )

        return self.precision, self.recall, self.Fbeta
