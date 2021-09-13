"""General purpose (impeded or unimpeded) data engineering nodes
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any


def add_train_test_group_random(
    data: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> pd.DataFrame:
    # Set random seed
    np.random.seed(random_seed)
    # Apply group
    data['group'] = data.apply(
        lambda row: 'test' if np.random.uniform() < test_size else 'train',
        axis=1,
    )

    return data


def add_train_test_group_by_time(
    data: pd.DataFrame,
    test_set_start_time: datetime,
) -> pd.DataFrame:
    data['group'] = data.apply(
        lambda row: 'test' if row['arrival_runway_actual_time'] >= test_set_start_time else 'train',
        axis=1,
    )

    return data


def add_train_test_group(
    data: pd.DataFrame,
    parameters: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    if isinstance(parameters['globals']['test_set_start_time'], datetime):
        # If a datetime test_set_start_time is specified in parameters, use it
        log.info(
            'assigning train & test groups based on test_set_start_time' +
            'in globals.yml'
        )
        data = add_train_test_group_by_time(
            data,
            parameters['globals']['test_set_start_time'],
        )
    else:
        # If no valid datetime TEST_SET_START_TIME in parameters,
        # default to random
        log.info(
            'assigning train & test groups randomly ' +
            'and based on TEST_SIZE and RANDOM_SEED parameters'
        )
        data = add_train_test_group_random(
            data,
            parameters['TEST_SIZE'],
            parameters['RANDOM_SEED']
        )

    return data


def train_test_group_logging(
    data: pd.DataFrame,
):
    log = logging.getLogger(__name__)

    log.info('train group has {} instances'.format(sum(data.group == 'train')))
    log.info('test group has {} instances'.format(sum(data.group == 'test')))


def set_index(
    data: pd.DataFrame,
    new_index="gufi",
) -> pd.DataFrame:
    data.set_index(new_index, inplace=True)

    return data


def keep_second_of_duplicate_gufi(
        data: pd.DataFrame,
        ) -> pd.DataFrame:
    data_out = data.loc[~data.gufi.duplicated(keep='first')]
    return data_out


def compute_total_taxi_time(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data['actual_arrival_total_taxi_time'] = \
        (
            data.arrival_stand_actual_time -
            data.arrival_runway_actual_time
        ).dt.total_seconds()

    return data


def left_join_on_index(
    data_0: pd.DataFrame,
    data_1: pd.DataFrame,
) -> pd.DataFrame:
    return data_0.join(data_1)


def replace_runway_actuals(
    data: pd.DataFrame,
    data_runway_actuals: pd.DataFrame,
) -> pd.DataFrame:
    # Replace arrival/departure runway, on times info at selected airport (arrival/departure times at other airports not modified)
    data_runway_actuals['isdeparture'] = data_runway_actuals['departure_runway_actual'].isna() == False
    data_runway_actuals['isarrival'] = data_runway_actuals['arrival_runway_actual'].isna() == False

    on_fields = ['gufi', 'isarrival', 'isdeparture']
    update_fields = [v for v in list(data_runway_actuals) if v not in on_fields]
    new_suffix = '_new'
    data_merged = pd.merge(data, data_runway_actuals, on=on_fields, how='left', suffixes=['', new_suffix], sort=False)

    for field in update_fields:
        bIndex = data_merged[field + new_suffix].isna() == False
        data_merged.loc[bIndex, field] = data_merged.loc[bIndex, field + new_suffix]
        if '_time' in field:  # Is this a sufficient check?
            data_merged[field] = pd.to_datetime(data_merged[field])

    data_merged.drop(columns=[v + new_suffix for v in update_fields], inplace=True)

    # Update derived fields
    data_merged['actual_arrival_ama_taxi_time'] = (data_merged['arrival_movement_area_actual_time'] - data_merged['arrival_runway_actual_time']).astype('timedelta64[s]')

    return data_merged

def merge_STBO(data: pd.DataFrame, data_STBO: pd.DataFrame
) -> pd.DataFrame:

    log = logging.getLogger(__name__)

    if data_STBO.empty:
        log.info('No STBO data available.')
        return data
    else:
        on_fields = ['acid', 'isarrival', 'isdeparture', 'departure_stand_initial_time', 'departure_aerodrome_icao_name', 'arrival_aerodrome_icao_name']
        new_suffix = '_ffs'

        merged_STBO_data = pd.merge(data, data_STBO, on = on_fields, how = 'left', suffixes =['', new_suffix])

        log.info('{:.1f}% of flights have STBO data elements available'.format(
        (merged_STBO_data['gufi_ffs'].notnull().sum()/len(merged_STBO_data))*100))
    
        return merged_STBO_data


def apply_filter_only_arrivals(
    data: pd.DataFrame,
) -> pd.DataFrame:
    initial_row_count = data.shape[0]

    data = data[data.isarrival]

    final_row_count = data.shape[0]

    log = logging.getLogger(__name__)
    log.info('Kept {:.1f}% of flights when filtering to keep only arrivals'.format(
        (final_row_count/initial_row_count)*100
    ))

    return data

def handle_round_robin_flights(
    data: pd.DataFrame,
) -> pd.DataFrame:
    # Round Robin flights have same departure/arrival airport, and get marked
    # with both isarrival and isdeparture as TRUE. However, some of these may
    # not have times for either the arrival or departure leg of the flight, 
    # and therefore we should change the isarrival or isdeparture flag to
    # FALSE
    
    log = logging.getLogger(__name__)
    
    current_arrival_count = sum(data['isarrival'])
    arr_index = ((data['isarrival'])&
                 (pd.isna(data['arrival_runway_actual_time']))&
                 (pd.isna(data['arrival_movement_area_actual_time']))&
                 (pd.isna(data['arrival_stand_actual_time'])))
    if sum(arr_index) > 0:
        data.loc[arr_index,'isarrival'] = False
    log.info('Changed {} isarrival flights to False'.format(
        current_arrival_count - sum(data['isarrival'])))
    
    current_departure_count = sum(data['isdeparture'])
    dep_index = ((data['isdeparture'])&
                 (pd.isna(data['departure_runway_actual_time']))&
                 (pd.isna(data['departure_movement_area_actual_time']))&
                 (pd.isna(data['departure_stand_actual_time'])))
    if sum(dep_index) > 0:
        data.loc[dep_index,'isdeparture'] = False
    log.info('Changed {} isdeparture flights to False'.format(
        current_departure_count - sum(data['isdeparture'])))
    
    return data

def coalesce_actual_times(
    data: pd.DataFrame,
) -> pd.DataFrame:
    '''
    This function will perform coalesce on the following fields:
    arrival_stand_actual_time = COALESCE(
                   arrival_stand_actual_time,
                   arrival_stand_airline_time)
     departure_stand_actual_time = COALESCE(
                   departure_stand_actual_time,
                   departure_stand_airline_time)
     
For each of these fields, create a new field that copies the original actual
times, so that we can use them later if needed
    '''
    
    # Make copies of originals
    data['arrival_stand_actual_time_orig'] = data['arrival_stand_actual_time']
    data['departure_stand_actual_time_orig'] = data['departure_stand_actual_time']
    
    # Coalesce New Times
    data['arrival_stand_actual_time'] = data['arrival_stand_actual_time'].combine_first(
        data['arrival_stand_airline_time'])
    data['departure_stand_actual_time'] = data['departure_stand_actual_time'].combine_first(
        data['departure_stand_airline_time'])
    
    return data

    