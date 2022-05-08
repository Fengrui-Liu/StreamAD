from os.path import dirname, join
import numpy as np
import pandas as pd


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
        self.data = pd.read_csv(self.path)
        self.names = self.data.columns.values

    def preprocess_timestamp(self) -> None:
        if "timestamp" in self.names:
            self.date = self.data["timestamp"].values
        else:
            self.date = self.data.index.values

    def preprocess_label(self) -> None:
        if "label" in self.names:
            self.label = np.array(self.data["label"].values)
        else:
            self.label = None

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
    """

    def __init__(self, f_path) -> None:
        super().__init__()
        self.path = f_path
        self.preprocess()
