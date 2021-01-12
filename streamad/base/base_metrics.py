from abc import ABC, ABCMeta, abstractmethod


class BaseMetrics(ABC):
    """
    Abstract class for evaluation metrics, supporting for customize evaluation.
    """

    def __init__(self) -> None:
        super().__init__()

        self.score = None
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        if y_pred == None:
            pass
        else:
            self.y_true.append(y_true)
            self.y_pred.append(y_pred)

    def reset(self):
        self.y_true = []
        self.y_pred = []

    @abstractmethod
    def evaluate(self):

        return 0.0