# StreamAD

![StreamAD Logo](docs/source/images/logo_htmlwithname.svg)





Anomaly detection for data streams/time series. Detectors process the univariate or multivariate data one by one to simulte a real-time scene.



[Documentation](https://streamad.readthedocs.io/en/latest/)


<!--- BADGES: START --->

![GitHub](https://img.shields.io/github/license/Fengrui-Liu/StreamAD)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/StreamAD?style=flat)
![Read the Docs](https://img.shields.io/readthedocs/streamad?style=flat)
![PyPI](https://img.shields.io/pypi/v/streamad)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/streamad)
[![Downloads](https://static.pepy.tech/personalized-badge/streamad?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/streamad)
![example workflow](https://github.com/Fengrui-Liu/StreamAD/actions/workflows/testing.yml//badge.svg)
[![codecov](https://codecov.io/gh/Fengrui-Liu/StreamAD/branch/main/graph/badge.svg?token=AQG26L2RA7)](https://codecov.io/gh/Fengrui-Liu/StreamAD)
![Conda](https://img.shields.io/conda/v/conda-forge/streamad)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FFengrui-Liu%2FStreamAD.svg?type=small)](https://app.fossa.com/projects/git%2Bgithub.com%2FFengrui-Liu%2FStreamAD?ref=badge_small)

---



## Installation

The stable version can be installed from PyPI:

```bash
pip install streamad
```

The development version can be installed from GitHub:

```bash
pip install git+https://github.com/Fengrui-Liu/StreamAD
```



## MODELS

### For univariate time series


| Model Example                                                                                                     | API Usage                                                                                                     | Paper                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [KNNCAD](https://streamad.readthedocs.io/en/latest/example/univariate.html#knncad-detector)                       | [streamad.model.KNNDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#knndetector)   | [Conformalized density- and distance-based anomaly detection in time-series data](https://arxiv.org/abs/1608.04585) |
| [SPOT](https://streamad.readthedocs.io/en/latest/example/univariate.html#spot-detector)                           | [streamad.model.SpotDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#spotdetector) | [Anomaly detection in streams with extreme value theory](https://dl.acm.org/doi/10.1145/3097983.3098144)            |
| [RRCF](https://streamad.readthedocs.io/en/latest/example/univariate.html#rrcf-detector)                           | [streamad.model.RrcfDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#rrcfdetector) | [Robust random cut forest based anomaly detection on streams](http://proceedings.mlr.press/v48/guha16.pdf)          |
| [Spectral Residual](https://streamad.readthedocs.io/en/latest/example/univariate.html#spectral-residual-detector) | [streamad.model.SRDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#srdetector)     | [Time-series anomaly detection service at microsoft](https://arxiv.org/abs/1906.03821)                              |


### For multivariate time series, also compatible with univariate time series.

| Models Example                                                                                  | API Usage                                                                                                          | Paper                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [xStream](https://streamad.readthedocs.io/en/latest/example/multivariate.html#xstream-detector) | [streamad.model.xStramDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#xstreamdetector) | [Xstream: outlier detection in feature-evolving data streams](http://www.kdd.org/kdd2018/accepted-papers/view/xstream-outlier-detection-in-feature-evolving-data-streams) |
| [RShash](https://streamad.readthedocs.io/en/latest/example/multivariate.html#rshash-detector)   | [streamad.model.RShashDetector](https://streamad.readthedocs.io/en/latest/api/streamad.model.html#rshashdetector)  | [Subspace Outlier Detection in Linear Time with Randomized Hashing](https://ieeexplore.ieee.org/document/7837870)                                                         |
