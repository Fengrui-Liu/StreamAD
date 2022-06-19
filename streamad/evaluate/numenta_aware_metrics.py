from streamad.base import BaseMetrics
from streamad.evaluate.ts_metrics import TSMetric
import numpy as np


class NumentaAwareMetircs(BaseMetrics):
    def __init__(self, anomaly_threshold: float = 0.8, beta: float = 1.0):
        """Numenta metrics calculation methods. :cite:`DBLP:journals/ijon/AhmadLPA17`.

        Args:
            anomaly_threshold (float, optional): A threshold to determine the anomalies, it can covert the anomaly scores to binary (0/1) indicators. Defaults to 0.8.
            beta (float, optional): F-beta score, like a F1-score. Defaults to 1.0.
        """
        super().__init__()
        self.threshold = anomaly_threshold
        self.beta = beta
        self.precision = None
        self.recall = None
        self.Fbeta = None

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        super().evaluate(y_true, y_pred)

        select = self.y_pred > self.threshold
        self.y_pred[select] = 1
        self.y_pred[~select] = 0

        metric = TSMetric(
            metric_option="numenta",
            beta=self.beta,
            alpha_r=0.0,
            cardinality="one",
            bias_p="flat",
            bias_r="flat",
        )
        self.precision, self.recall, self.Fbeta = metric.score(
            self.y_true, self.y_pred
        )

        return self.precision, self.recall, self.Fbeta
