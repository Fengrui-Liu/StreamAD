from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import KNNDetector
from streamad.process import ZScoreCalibrator, TDigestCalibrator


def test_ZScoreCalibrator():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = KNNDetector()
    calibrator = ZScoreCalibrator(sigma=2, extreme_sigma=3)

    for x in stream.iter_item():
        score = detector.fit_score(x)
        score = calibrator.normalize(score)
        if score is not None:
            assert 0 <= score <= 1

def test_ZScoreCalibrator_global():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = KNNDetector()
    calibrator = ZScoreCalibrator(sigma=2, is_global=True)

    for x in stream.iter_item():
        score = detector.fit_score(x)
        score = calibrator.normalize(score)
        if score is not None:
            assert 0 <= score <= 1


def test_TDigestCalibrator():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = KNNDetector()
    calibrator = TDigestCalibrator(percentile_up=93, percentile_down=0)

    for x in stream.iter_item():
        score = detector.fit_score(x)
        normalized_score = calibrator.normalize(score)
        if normalized_score is not None:
            assert 0 <= normalized_score <= 1


def test_TDigestCalibrator_global():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    detector = KNNDetector()
    calibrator = TDigestCalibrator(
        percentile_up=93, percentile_down=0, is_global=True
    )

    for x in stream.iter_item():
        score = detector.fit_score(x)
        score = calibrator.normalize(score)
        if score is not None:
            assert 0 <= score <= 1
