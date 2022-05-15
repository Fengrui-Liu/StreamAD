import warnings
from os.path import dirname, join
from typing import Union

import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)


class DS:
    def __init__(self) -> None:

        self.data = None
        self.date = None
        self.label = None
        self.features = None
        self.names = None

    def preprocess(self) -> None:
        self.preprocess_data()
        self.preprocess_timestamp()
        self.preprocess_label()
        self.preprocess_feature()

    def preprocess_data(self) -> None:
        if type(self.path) == str:
            try:
                self.data = pd.read_csv(self.path)
            except FileExistsError:
                print("Cannot read this file:", self.path)
        elif type(self.path) == np.ndarray:
            self.data = pd.DataFrame(self.path)
        elif type(self.path) == pd.DataFrame:
            self.data = self.path
        self.names = self.data.columns.values

    def preprocess_timestamp(self) -> None:
        if "timestamp" in self.names.tolist():
            self.date = self.data["timestamp"].values
        else:
            self.date = self.data.index.values

    def preprocess_label(self) -> None:
        if "label" in self.names.tolist():
            self.label = np.array(self.data["label"].values)

    def preprocess_feature(self) -> None:
        self.features = np.setdiff1d(
            self.names, np.array(["label", "timestamp"])
        )
        self.data = np.array(self.data[self.features])


class MultivariateDS(DS):
    """
    Load multivariate dataset.
    """

    def __init__(self, has_names=False) -> None:
        super().__init__()
        module_path = dirname(__file__)
        self.path = join(module_path, "data", "multiDS.csv")
        self.preprocess()


class UnivariateDS(DS):
    """
    Load univariate dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        module_path = dirname(__file__)
        self.path = join(module_path, "data", "uniDS.csv")
        self.preprocess()


class CustomDS(DS):
    """
    Load custom dataset.
    Args:
        f_path (Union[str, np.ndarray]): Dataset or its path.
        label (np.ndarray, optional): Anomaly labels for dataset. Defaults to None.
    """

    def __init__(
        self, f_path: Union[str, np.ndarray], label: np.ndarray = None
    ):

        super().__init__()
        self.path = f_path
        self.label = label
        self.preprocess()
