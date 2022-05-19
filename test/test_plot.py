import numpy as np
from streamad.util import StreamGenerator, CustomDS, plot
from streamad.model import ZScoreDetector


def test_plot():
    n, A, center, phi = 730, 50, 100, 30
    T = 2 * np.pi / 100
    t = np.arange(n)
    ds = A * np.sin(T * t - phi * T) + center
    ds[235:255] = 80
    label = np.array([0] * n)
    label[235:255] = 1

    ds = CustomDS(ds, label)  # You can also use a file path here
    stream = StreamGenerator(ds.data)
    model = ZScoreDetector()

    scores = []

    for x in stream.iter_item():
        score = model.fit_score(x)
        scores.append(score)
        # print("\r Anomaly score: {}".format(score), end="", flush="True")

    data, label, date, features = ds.data, ds.label, ds.date, ds.features
    plot(data=data, scores=scores, date=date, features=features, label=label)
