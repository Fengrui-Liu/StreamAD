from streamad.model import SpectralResidual
from sklearn.metrics import roc_curve, auc
import random
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "../..")))

from streamad.util import StreamGenerator, StreamStatistic

from streamad.util import UnivariateDS, MultivariateDS
import matplotlib.pylab as plt


# test_ts needs to import timesynth, to install you can use "pip install timesynth"

# import timesynth as ts

# def test_ts():
#     od = SpectralResidual()
#     n_points = 10000
#     time_sampler = ts.TimeSampler(stop_time=n_points // 4)
#     time_samples = time_sampler.sample_regular_time(num_points=n_points)

#     # harmonic time series with Gaussian noise
#     sinusoid = ts.signals.Sinusoidal(frequency=0.25)
#     white_noise = ts.noise.GaussianNoise(std=0.1)
#     ts_harm = ts.TimeSeries(signal_generator=sinusoid, noise_generator=white_noise)
#     samples, signals, errors = ts_harm.sample(time_samples)
#     X = samples.reshape(-1, 1).astype(np.float32)

#     data = inject_outlier_ts(X, perc_outlier=10, perc_window=10, n_std=2.0, min_std=1.0)
#     X_outlier, y_outlier, labels = data.data, data.target.astype(int), data.target_names

#     for X in X_outlier:
#         od.fit(X)
#     y_score = od.score(X_outlier, 1)

#     fpr, tpr, threshold = roc_curve(y_outlier[-10000:], y_score)
#     roc_auc = auc(fpr, tpr)  ###计算auc的值
#     assert roc_auc > 0.8


class Bunch(dict):
    """
    Container object for internal datasets
    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def inject_outlier_ts(
    X: np.ndarray,
    perc_outlier: int,
    perc_window: int = 10,
    n_std: float = 2.0,
    min_std: float = 1.0,
) -> Bunch:
    """
    Inject outliers in both univariate and multivariate time series data.
    Parameters
    ----------
    X
        Time series data to perturb (inject outliers).
    perc_outlier
        Percentage of observations which are perturbed to outliers. For multivariate data,
        the percentage is evenly split across the individual time series.
    perc_window
        Percentage of the observations used to compute the standard deviation used in the perturbation.
    n_std
        Number of standard deviations in the window used to perturb the original data.
    min_std
        Minimum number of standard deviations away from the current observation. This is included because
        of the stochastic nature of the perturbation which could lead to minimal perturbations without a floor.
    Returns
    -------
    Bunch object with the perturbed time series and the outlier labels.
    """
    n_dim = len(X.shape)
    if n_dim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_ts = X.shape
    X_outlier = X.copy()
    is_outlier = np.zeros(n_samples)
    # one sided window used to compute mean and stdev from
    window = int(perc_window * n_samples * 0.5 / 100)
    # distribute outliers evenly over different time series
    n_outlier = int(n_samples * perc_outlier * 0.01 / n_ts)
    if n_outlier == 0:
        return Bunch(
            data=X_outlier, target=is_outlier, target_names=["normal", "outlier"]
        )
    for s in range(n_ts):
        outlier_idx = np.sort(random.sample(range(n_samples), n_outlier))
        window_idx = [
            np.maximum(outlier_idx - window, 0),
            np.minimum(outlier_idx + window, n_samples),
        ]
        stdev = np.array(
            [
                X_outlier[window_idx[0][i] : window_idx[1][i], s].std()
                for i in range(len(outlier_idx))
            ]
        )
        rnd = np.random.normal(size=n_outlier)
        X_outlier[outlier_idx, s] += (
            np.sign(rnd) * np.maximum(np.abs(rnd * n_std), min_std) * stdev
        )
        is_outlier[outlier_idx] = 1
    if n_dim == 1:
        X_outlier = X_outlier.reshape(
            n_samples,
        )
    return Bunch(data=X_outlier, target=is_outlier, target_names=["normal", "outlier"])


def test_StreamGenerator():
    ds = UnivariateDS()
    data = ds.data
    label = ds.label
    stream = StreamGenerator(data, label, shuffle=False)
    od = SpectralResidual()

    # can fit data and get score one-by-one
    for X, z in stream.iter_item():
        od.fit(X)
    y_score = od.score(stream, 1)

    fpr, tpr, threshold = roc_curve(label, y_score)
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    assert roc_auc > 0.8
