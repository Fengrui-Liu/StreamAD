import numpy as np


class WeightEnsemble:
    def __init__(self, ensemble_weights: list = None):
        """Anomaly scores ensemble with weighted average.

        Args:
            ensemble_weights (list, optional): Weights for scores with orders, we use equal weights/mean to recalculate the scores when it is None. Defaults to None.
        """

        assert (
            type(ensemble_weights) == list
            or type(ensemble_weights) == np.ndarray
        )

        self.weights = ensemble_weights
        self.sum_weights = np.sum(self.weights) if ensemble_weights else None

    def ensemble(self, scores: list) -> float:
        """Ensemble anomaly scores from ordered detectors.

        Args:
            scores (list): A list of anomaly scores with orders.

        Returns:
            float: Ensembled anomaly scores.
        """

        assert (
            type(scores) == list or type(scores) == np.ndarray
        ), "Unsupport score types, it should be list or numpy.ndarray"

        assert len(scores) == len(
            self.weights
        ), "Inconsistent weights and scores length"

        if (np.array(scores) == None).any():
            return None

        assert (
            (np.array(scores) >= 0) & (np.array(scores) <= 1)
        ).all(), (
            "Scores should be in [0,1], you can call calibrator before ensemble"
        )

        if self.weights is None:
            return np.mean(scores)

        return np.dot(scores, self.weights) / self.sum_weights
