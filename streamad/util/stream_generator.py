from typing import Generator

import numpy as np


class StreamGenerator:
    """Load static dataset and generate observation once a time.

    Args:
        X (np.ndarray): Origin static dataset.

    Raises:
        TypeError: Unexpected input data type.
    """

    def __init__(
        self, X: np.ndarray,
    ):

        if isinstance(X, np.ndarray):
            self.X = X
        else:
            raise TypeError("Unexpected input data type, except np.ndarray.")

    def iter_item(self) -> Generator:
        """Iterate item once a time from the dataset.

        Yields:
            Generator: One observation from the dataset.
        """

        for i in range(len(self.X)):
            yield self.X[i]
