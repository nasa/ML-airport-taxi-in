"""Unimpeded taxi-in data engineering nodes
"""

import pandas as pd

from typing import Dict, Any
import logging

def apply_filter_null_times(
    data: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    null_times = ((pd.isna(data['arrival_movement_area_actual_time']))|
                  (pd.isna(data['arrival_runway_actual_time']))|
                  (pd.isna(data['arrival_stand_actual_time'])))
    data = data.drop(data.index[null_times])

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Kept {:.1f}% of flights when filtering to keep only arrivals with non-nulls for all required actual times'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data


def apply_filter_req_arr_stand_and_runway(
    data: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    # Remove rows with no stand
    data = data[
        data.arrival_stand_actual.notnull()
    ]

    interim_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Kept {:.1f}% of arrivals when filtering to keep only arrivals with non-null arrival stand'.format(
        (interim_row_count/initial_row_count)*100
    ))

    # Remove rows with no runway
    data = data[
        data.arrival_runway_actual.notnull()
    ]

    final_row_count = data.shape[0]

    log.info('Kept {:.1f}% of arrivals when filtering to keep only arrivals with non-null arrival runway'.format(
        (final_row_count/interim_row_count)*100
    ))

    return data


def apply_filter_req_arr_taxi_times(
    data: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    # Remove rows if 0 or negative total or AMA or ramp taxi times
    # data = data[
    #     data.actual_arrival_total_taxi_time > 0.0
    # ]
    data = data[
        ((data.actual_arrival_total_taxi_time > 0.0)&(data.actual_arrival_total_taxi_time < 3600))
    ]

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Kept {:.1f}% of arrivals when filtering to keep only arrivals with valid arrival total taxi time'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data


def apply_filter_req_arr_ramp_taxi_times(
    data: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    # Remove rows if 0 or negative total or AMA or ramp taxi times
    data = data[
        data.actual_arrival_ramp_taxi_time > 0.0
    ]

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Kept {:.1f}% of arrivals when filtering to keep only arrivals with valid arrival ramp taxi time'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data


def apply_filter_req_arr_ama_taxi_times(
    data: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    # Remove rows if 0 or negative total or AMA or ramp taxi times
    data = data[
        ((data.actual_arrival_ama_taxi_time > 0.0)&(data.actual_arrival_ama_taxi_time < 3600))
    ]

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Kept {:.1f}% of arrivals when filtering to keep only arrivals with valid arrival AMA taxi time'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data


def join_fraction_speed_gte_threshold_and_filter(
    data: pd.DataFrame,
    data_fraction_speed_gte_threshold: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    data = data.join(data_fraction_speed_gte_threshold)

    data = data[data.fraction_speed_gte_threshold.notnull()]

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Matched {:.1f}% of arrivals when joining fraction of taxi-in trajectory with speed greater than or equal to threshold'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data


def calculate_unimpeded_AMA(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
):
    data['unimpeded_AMA'] = (
        data.fraction_speed_gte_threshold >=
        model_params['unimpeded_ama_fraction_gte']
    )

    log = logging.getLogger(__name__)
    log.info('{:.1f}% of arrivals were unimpeded in the AMA (because fraction of trajectory measurements with speed above threshold >= {})'.format(
        data.unimpeded_AMA.mean()*100,
        model_params['unimpeded_ama_fraction_gte'],
    ))

    return data
