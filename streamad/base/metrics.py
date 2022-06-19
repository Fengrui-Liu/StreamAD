from abc import ABC, abstractmethod
import numpy as np


class BaseMetrics(ABC):
    """
    Abstract class for evaluation metrics, supporting for customize evaluation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.y_pred = None
        self.y_true = None

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_pred = np.array(y_pred)
        y_pred[y_pred == None] = 0
        self.y_true = y_true.astype(int)
        self.y_pred = y_pred.astype(int)
        return
