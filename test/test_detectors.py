from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import (
    KNNDetector,
    SpotDetector,
    RandomDetector,
    xStreamDetector,
)


uni_detectors = {
    KNNDetector: {},
    SpotDetector: {},
    RandomDetector: {},
    xStreamDetector: {},
}

multi_detectors = {
    RandomDetector: {},
    xStreamDetector: {},
}


detectors_lst = []


def helper_init_detector(detector, params):
    model = detector(**params)

    return model


def test_univariate_ts():
    detectors = []

    for detector, params in uni_detectors.items():
        if type(params) is dict:
            d = helper_init_detector(detector, params)
            detectors.append(d)

        if type(params) is list:
            for param in params:
                d = helper_init_detector(detector, param)
                detectors.append(d)

    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)

    for x in stream.iter_item():
        for detector in detectors:
            detector.fit_score(x)
