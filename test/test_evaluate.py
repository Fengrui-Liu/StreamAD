from streamad.util import StreamGenerator, UnivariateDS
from streamad.model import ZScoreDetector
from streamad.evaluate import AUCMetric


def test_AUC():
    ds = UnivariateDS()
    stream = StreamGenerator(ds.data)
    model = ZScoreDetector()

    scores = []

    for x in stream.iter_item():
        score = model.fit_score(x)
        scores.append(score)
        # print("\r Anomaly score: {}".format(score), end="", flush="True")

    label = ds.label
    auc_score = AUCMetric().evaluate(label, scores)
