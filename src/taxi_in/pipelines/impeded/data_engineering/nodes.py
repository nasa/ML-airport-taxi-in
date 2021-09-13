import pandas as pd
from typing import Any, Dict
import logging

def add_scheduled_taxi_in_predictions(
        data: pd.DataFrame,
        scheduled_taxi_in_predictions: pd.DataFrame,
) -> pd.DataFrame:
    
    data = pd.merge(data,
                    scheduled_taxi_in_predictions,
                    on = 'gufi',
                    how = 'left')
    
    return data

def add_arrival_stand_airline_time_at_landing(
        data: pd.DataFrame,
        arrival_stand_airline_time_at_landing: pd.DataFrame,
) -> pd.DataFrame:
    
    data = pd.merge(data,
                    arrival_stand_airline_time_at_landing[
                        ['gufi',
                         'arrival_runway_airline_time_at_landing', 
                         'arrival_stand_airline_time_at_landing',
                         'airline_taxi_in_prediction_seconds']],
                    on = 'gufi',
                    how = 'left')
    
    return data

def prepare_aircraft_class_map(
    aircraft_categories: pd.DataFrame,
) -> Dict[str, str]:
    # prepare aircraft class data
    aircraft_categories.rename(
        columns={
            "aircraft_type": 'aircraft_type',
            "category": "aircraft_category"

        },
        inplace=True
    )
    aircraft_categories.aircraft_type = aircraft_categories.aircraft_type.astype(str)
    aircraft_categories.set_index('aircraft_type', inplace=True)
    aircraft_categories = aircraft_categories.squeeze().to_dict()

    return aircraft_categories

def apply_imp_filters(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    # Remove rows with no stand
    data = data[
        data.arrival_stand_actual.notnull()
    ]

    # Remove rows with no runway
    data = data[
        data.arrival_runway_actual.notnull()
    ]

    data = data[
        data.actual_arrival_ama_taxi_time > 0.0
    ]
    data = data[
        data.actual_arrival_ramp_taxi_time > 0.0
    ]

    # Remove rows with missing features
    data.dropna(
        axis=0,
        how='any',
        subset=model_params['features'] + [
            model_params['target'],
            'group',
        ],
        inplace=True,
    )

    final_row_count = data.shape[0]
    
    log = logging.getLogger(__name__)
    log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ... kept {:.1f}% of flights'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data
