import numpy as np
from streamad.evaluate import (
    NumentaAwareMetircs,
    PointAwareMetircs,
    SeriesAwareMetircs,
)


def test_point_aware_metrics():
    values_real = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    values_pred = np.array([0, 0, 0, None, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    metric = PointAwareMetircs(anomaly_threshold=0.8)

    (precision, recall, f1,) = metric.evaluate(values_real, values_pred)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_series_aware_metrics():
    values_real = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    values_pred = np.array([0, 0, 0, None, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    # Flat bias
    metric = SeriesAwareMetircs(
        anomaly_threshold=0.8, bias_p="flat", bias_r="flat"
    )

    (precision, recall, f1,) = metric.evaluate(values_real, values_pred)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0

    # Front bias
    metric = SeriesAwareMetircs(
        anomaly_threshold=0.8, bias_p="flat", bias_r="front"
    )

    (precision, recall, f1,) = metric.evaluate(values_real, values_pred)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0

    # Middle bias
    metric = SeriesAwareMetircs(
        anomaly_threshold=0.8, bias_p="flat", bias_r="middle"
    )

    (precision, recall, f1,) = metric.evaluate(values_real, values_pred)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0

    # Back bias
    metric = SeriesAwareMetircs(
        anomaly_threshold=0.8, bias_p="flat", bias_r="back"
    )

    (precision, recall, f1,) = metric.evaluate(values_real, values_pred)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_numenta_aware_metrics():
    values_real = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    values_pred = np.array([0, 0, 0, None, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    metric = NumentaAwareMetircs(anomaly_threshold=0.8)

    (precision, recall, f1,) = metric.evaluate(values_real, values_pred)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
