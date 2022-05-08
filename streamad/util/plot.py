import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot(data, label, date, scores, features):

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

    anomalies = date[np.where(label == 1)[0]]

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Scatter(x=date, y=data[:, i], name=feature), row=i + 1, col=1
        )

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
    fig.show()
