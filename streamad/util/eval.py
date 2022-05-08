from sklearn.metrics import roc_auc_score
from streamad.base import BaseMetrics


class AUCMetric(BaseMetrics):
    """ROC_AUC score for evaluation."""

    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, y_true, y_pred):

        return roc_auc_score(y_true, y_pred)
