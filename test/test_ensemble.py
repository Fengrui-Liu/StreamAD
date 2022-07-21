from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import KNNDetector, SpotDetector
from streamad.process import ZScoreCalibrator, VoteEnsemble, WeightEnsemble


def test_VoteEnsemble():

    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    knn_detector = KNNDetector()
    spot_detector = SpotDetector()
    knn_calibrator = ZScoreCalibrator(sigma=2)
    spot_calibrator = ZScoreCalibrator(sigma=2)
    ensemble = VoteEnsemble(threshold=0.8)

    for x in stream.iter_item():

        knn_score = knn_detector.fit_score(x)
        spot_score = spot_detector.fit_score(x)

        knn_normalized_score = knn_calibrator.normalize(knn_score)
        spot_normalized_score = spot_calibrator.normalize(spot_score)

        score = ensemble.ensemble([knn_normalized_score, spot_normalized_score])
        if score is not None:
            assert 0 <= score <= 1


def test_WeightEnsemble():

    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    knn_detector = KNNDetector()
    spot_detector = SpotDetector()
    knn_calibrator = ZScoreCalibrator(sigma=3)
    spot_calibrator = ZScoreCalibrator(sigma=3)
    ensemble = WeightEnsemble(ensemble_weights=[0.6, 0.4])

    for x in stream.iter_item():
        knn_score = knn_detector.fit_score(x)
        spot_score = spot_detector.fit_score(x)

        knn_normalized_score = knn_calibrator.normalize(knn_score)
        spot_normalized_score = spot_calibrator.normalize(spot_score)

        score = ensemble.ensemble([knn_normalized_score, spot_normalized_score])

        if score is not None:
            assert 0 <= score <= 1
