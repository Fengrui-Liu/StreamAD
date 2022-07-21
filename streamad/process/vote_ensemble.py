import numpy as np


class VoteEnsemble:
    def __init__(self, threshold: float = 0.8):
        """Anomaly scores ensemble with votes.

        Args:
            threshold (float, optional): Anomaly scores that over threshold are regard as votes. Defaults to 0.8.
        """
        self.thredshold = threshold

    def ensemble(self, scores: list):
        """Ensemble anomaly scores from ordered detectors.

        Args:
            scores (list): A list of anomaly scores with orders.

        Returns:
            float: Ensembled anomaly scores.
        """

        assert (
            type(scores) == list or type(scores) == np.ndarray
        ), "Unsupport score types, it should be list or numpy.ndarray"

        if (np.array(scores) == None).any():
            return None

        assert (
            (np.array(scores) >= 0) & (np.array(scores) <= 1)
        ).all(), (
            "Scores should be in [0,1], you can call calibrator before ensemble"
        )

        votes = np.array(scores) >= self.thredshold

        if sum(votes) > len(votes) / 2:
            return 1.0

        return 0.0
