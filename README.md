# StreamAD

![StreamAD Logo](docs/source/images/logo_htmlwithname.svg)



Anomaly detection for data streams/time series. Detectors process the univariate or multivariate data one by one to simulte a real-time scene.



[Documentation](https://streamad.readthedocs.io/en/latest/)


<!--- BADGES: START --->



![Conda](https://img.shields.io/conda/v/conda-forge/streamad)
![PyPI](https://img.shields.io/pypi/v/streamad)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/StreamAD?style=flat)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/streamad)

![Read the Docs](https://img.shields.io/readthedocs/streamad?style=flat)
![GitHub](https://img.shields.io/github/license/Fengrui-Liu/StreamAD)
![Conda](https://img.shields.io/conda/pn/conda-forge/streamad)
[![Downloads](https://static.pepy.tech/personalized-badge/streamad?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/streamad)


![example workflow](https://github.com/Fengrui-Liu/StreamAD/actions/workflows/testing.yml//badge.svg)
[![codecov](https://codecov.io/gh/Fengrui-Liu/StreamAD/branch/main/graph/badge.svg?token=AQG26L2RA7)](https://codecov.io/gh/Fengrui-Liu/StreamAD)
[![Maintainability](https://api.codeclimate.com/v1/badges/525d7e3663ee4c5c0daa/maintainability)](https://codeclimate.com/github/Fengrui-Liu/StreamAD/maintainability)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FFengrui-Liu%2FStreamAD.svg?type=small)](https://app.fossa.com/projects/git%2Bgithub.com%2FFengrui-Liu%2FStreamAD?ref=badge_small)



---



## Installation

The stable version can be installed from PyPI:

```bash
pip install streamad
```

The stable version can be installed from conda-forge:

```bash
conda install --channel "conda-forge" streamad
```

The development version can be installed from GitHub:

```bash
pip install git+https://github.com/Fengrui-Liu/StreamAD
```

---

## Quick Start

Start once detection within 5 lines of code. You can find more example with visualization results [here](https://streamad.readthedocs.io/en/latest/example/example.html).

```python
from streamad.util import StreamGenerator, UnivariateDS, plot
from streamad.model import SpotDetector

ds = UnivariateDS()
stream = StreamGenerator(ds.data)
model = SpotDetector()

for x in stream.iter_item():
    score = model.fit_score(x)
```

## Models

### For univariate time series

If you want to detect multivarite time series with these models, you need to apply them on each feature separately.

| Model Example                                                                                                     | API Usage                                                                                                         | Paper                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [KNNCAD](https://streamad.readthedocs.io/en/latest/example/univariate.html#knncad-detector)                       | [streamad.model.KNNDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#knndetector)       | [Conformalized density- and distance-based anomaly detection in time-series data](https://arxiv.org/abs/1608.04585) |
| [SPOT](https://streamad.readthedocs.io/en/latest/example/univariate.html#spot-detector)                           | [streamad.model.SpotDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#spotdetector)     | [Anomaly detection in streams with extreme value theory](https://dl.acm.org/doi/10.1145/3097983.3098144)            |
| [RRCF](https://streamad.readthedocs.io/en/latest/example/univariate.html#rrcf-detector)                           | [streamad.model.RrcfDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#rrcfdetector)     | [Robust random cut forest based anomaly detection on streams](http://proceedings.mlr.press/v48/guha16.pdf)          |
| [Spectral Residual](https://streamad.readthedocs.io/en/latest/example/univariate.html#spectral-residual-detector) | [streamad.model.SRDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#srdetector)         | [Time-series anomaly detection service at microsoft](https://arxiv.org/abs/1906.03821)                              |
| [Z score](https://streamad.readthedocs.io/en/latest/example/univariate.html#z-score-detector)                     | [streamad.model.ZScoreDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#zscoredetector) | [Standard score](https://en.wikipedia.org/wiki/Standard_score)                                                      |


### For multivariate time series

These models are compatible with univariate time series.



| Models Example                                                                                         | API Usage                                                                                                          | Paper                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [xStream](https://streamad.readthedocs.io/en/latest/example/multivariate.html#xstream-detector)        | [streamad.model.xStramDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#xstreamdetector) | [Xstream: outlier detection in feature-evolving data streams](http://www.kdd.org/kdd2018/accepted-papers/view/xstream-outlier-detection-in-feature-evolving-data-streams) |
| [RShash](https://streamad.readthedocs.io/en/latest/example/multivariate.html#rshash-detector)          | [streamad.model.RShashDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#rshashdetector)  | [Subspace Outlier Detection in Linear Time with Randomized Hashing](https://ieeexplore.ieee.org/document/7837870)                                                         |
| [HSTree](https://streamad.readthedocs.io/en/latest/example/multivariate.html#half-space-tree-detector) | [streamad.model.HSTreeDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#hstreedetector)  | [Fast Anomaly Detection for Streaming Data](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)                                                                          |
| [LODA](https://streamad.readthedocs.io/en/latest/example/multivariate.html#loda-detector)              | [streamad.model.LodaDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#lodadetector)      | [Lightweight on-line detector of anomalies](https://link.springer.com/article/10.1007/s10994-015-5521-0)                                                                  |