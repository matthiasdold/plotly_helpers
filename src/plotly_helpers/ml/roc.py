from sklearn import metrics

import numpy as np

import plotly.graph_objects as go


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:

    fpr, tpr, ths = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)

    fig = go.Figure()
    fig.add_traces(
        [

            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line_color='#888888'),
            go.Scatter(x=fpr, y=tpr, mode='lines', line_color='#5555ff',
                       name='ROC', fill='tonexty'),
        ]
    )

    fig.add_annotation(x=0.5, y=0.8, text=f"AUC={auc:.1%}")

    fig.update_layout(xaxis={'title': 'False Positive Rate'},
                      yaxis={'title': 'True Positive Rate'})

    return fig


if __name__ == "__main__":
    y_true = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    y_pred = [0.8, 0.7, 0.3, 1.2, 1.1, 1.2, 0.8, 0.5, 1, 0.8]

    fig = plot_roc_curve(y_true, y_pred)
    fig.show()
