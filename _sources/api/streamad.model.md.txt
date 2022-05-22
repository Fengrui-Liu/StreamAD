
# StreamAD Detector


## Univariate Anomaly Detector

If you want to detect multivarite time series with these models, you need to apply them on each feature separately.
### KNNDetector

```{eval-rst}
.. autoclass:: streamad.model.KNNDetector
    :show-inheritance:
    :members: parse
```

----

### SpotDetector

```{eval-rst}
.. autoclass:: streamad.model.SpotDetector
    :show-inheritance:
    :members: parse
```


----


### RrcfDetector

```{eval-rst}
.. autoclass:: streamad.model.RrcfDetector
    :show-inheritance:
    :members: parse
```

----


### SRDetector

```{eval-rst}
.. autoclass:: streamad.model.SRDetector
    :show-inheritance:
    :members: parse
```

----


### ZScoreDetector

```{eval-rst}
.. autoclass:: streamad.model.ZScoreDetector
    :show-inheritance:
    :members: parse
```

----


## Multivariate Anomaly Detector

These models are compatible with univariate time series.

### xStreamDetector

```{eval-rst}
.. autoclass:: streamad.model.xStreamDetector
    :show-inheritance:
    :members: parse
```

----

### RShashDetector

```{eval-rst}
.. autoclass:: streamad.model.RShashDetector
    :show-inheritance:
    :members: parse
```

----

### HSTreeDetector

```{eval-rst}
.. autoclass:: streamad.model.HSTreeDetector
    :show-inheritance:
    :members: parse
```

----

### RandomDetector

```{eval-rst}
.. autoclass:: streamad.model.RandomDetector
    :show-inheritance:
    :members: parse
```
