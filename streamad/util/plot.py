import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot(
    data: np.ndarray,
    scores: np.ndarray,
    date: np.ndarray = None,
    features: np.ndarray = None,
    label: np.ndarray = None,
):
    """Plot data, score and ground truth (if exists).

    Args:
        data (np.array): Original data stream.
        scores (np.array): Anomaly scores of the data stream.
        date (np.array, optional): Timestamp of the data. Defaults to None.
        features (np.array, optional): Features name. Defaults to None.
        label (np.array, optional): Ground truth. Defaults to None.
    """

    if features is None:
        features = ["f" + str(i) for i in range(np.array(data).shape[1])]
    else:
        assert (
            len(features) == data.shape[1]
        ), "Number of features must match data dimension."

    if date is None:
        date = [i for i in range(np.array(data).shape[0])]
    else:
        assert (
            len(date) == data.shape[0]
        ), "Number of date must match data dimension."

    height = 100 * len(features) + 80
    row_heights = [100 / height for _ in range(len(features))]
    row_heights.append(80 / height)

    fig = make_subplots(
        rows=len(features) + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=20 / height,
        row_heights=row_heights,
    )

    # Plot data by features
    for i, feature in enumerate(features):
        fig.add_trace(
            go.Scatter(x=date, y=data[:, i], name=str(feature)),
            row=i + 1,
            col=1,
        )

    # Plot label by anomalies
    if label is not None:
        anomalies = date[np.where(label == 1)[0]]
        for anomaly in anomalies:
            fig.add_vrect(
                x0=anomaly,
                x1=anomaly,
                fillcolor="red",
                opacity=0.25,
                row="all",
                col=1,
                name="Anomaly",
            )

    # Plot score
    fig.add_trace(
        go.Scatter(x=date, y=scores, name="anomaly score", marker_color="red"),
        row=len(features) + 1,
        col=1,
    )
    # fig.update_xaxes(rangeslider={"visible": True}, row=2, col=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        height=height,
    )
    return fig
