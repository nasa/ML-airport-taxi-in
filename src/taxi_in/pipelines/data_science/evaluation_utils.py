"""
Functions that are helpful for evaluating models.
"""

import pandas as pd
from sklearn import metrics
from . import error_metrics


def evaluate_predictions(
    df,
    y_true,
    y_pred,
    metrics_dict={'mean_absolute_error': metrics.mean_absolute_error},
):
    evaluation_df = pd.DataFrame(
        index=df.group.unique(),
    )

    for metric_name, metric_func in metrics_dict.items():
        if metric_name == 'percent within n':
            continue  # Handled by separate function

        evaluation_df[metric_name] = None

        for group in df.group.unique():
            evaluation_df.loc[group, metric_name] =\
                metric_func(
                    df.loc[df.group == group, y_true],
                    df.loc[df.group == group, y_pred],
                )

    return evaluation_df


def calc_percent_within_n_df(
    df,
    y_true,
    y_pred,
    delta,
    max,
):
    evaluation_df = pd.DataFrame(
        index=df.group.unique(),
    )

    for t in range(delta, max+delta, delta):
        evaluation_df['percent_within_{}'.format(t)] = 0.0
        for group in df.group.unique():
            evaluation_df.loc[group, 'percent_within_{}'.format(t)] =\
                error_metrics.percent_within_n(
                    df.loc[df.group == group, y_true],
                    df.loc[df.group == group, y_pred],
                    t
                )

    return evaluation_df
