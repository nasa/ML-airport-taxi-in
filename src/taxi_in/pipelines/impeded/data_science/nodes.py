"""Data science nodes for impeded model development
"""

from typing import Any, Dict

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from data_services.mlflow_utils import add_environment_specs_to_conda_file
import numpy as np

import time
import mlflow
from mlflow import sklearn as mlf_sklearn
import logging

from data_services.stand_ama_avg_taxi_time_encoder import AvgStandAMATaxiEncoder
from data_services.stand_cluster_encoder import StandEncoder
from data_services.aircraft_class_encoder import AircraftClassEncoder
from data_services.runway_ama_avg_taxi_time_encoder import AvgRunwayAMATaxiEncoder
from data_services.ama_gate_avg_taxi_time_encoder import AvgRampTaxiEncoder
from data_services.FilterPipeline import FilterPipeline
from data_services.fill_nas import FillNAs
from data_services.OrderFeatures import OrderFeatures
from data_services.utils import FormatMissingData


def train_imp_AMA_model(
    aircraft_category: Dict[str,str],
    imp_data_engred: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,

) -> Pipeline:

    ac_enc=AircraftClassEncoder(
        aircraft_category,
    )

    stand_encoder = StandEncoder()

    runway_AMA_taxi_enc=AvgRunwayAMATaxiEncoder()

    stand_AMA_taxi_enc=AvgStandAMATaxiEncoder(  )

     # Build one-hot encoder for general categorical features

    oh_enc=OneHotEncoder(
         sparse=False,
         handle_unknown="ignore",
     )

     # Make column transformer
    col_transformer=ColumnTransformer(
        [
            ('stand_AMA_mean_taxi_time_encoder', stand_AMA_taxi_enc, ['arrival_runway_actual', 'arrival_stand_actual']),
            ('runway_AMA_mean_taxi_time_encoder', runway_AMA_taxi_enc, ['arrival_runway_actual']),
            ('aircraft_class_encoder', ac_enc, ['aircraft_type']),
            ('one_hot_encoder', oh_enc, model_params['OneHotEncoder_features']),
            ('stand_encoder', stand_encoder, ['arrival_stand_actual'])
        ],
         remainder='passthrough',
     )

    # Replaces miscellaneous missing values with expected values
    format_missing_data = FormatMissingData(model_params['features_core'])

    # Replace NAs with provided value, only affecting numeric/bool since categorical are handled at the encoder level
    fill_nas = FillNAs(0)

    # Orders feature columns
    order_features = OrderFeatures()

    #  Make model
    if model_params['model'] == 'RandomForestRegressor':
        imp_model = RandomForestRegressor(**model_params['model_params'])
    elif model_params['model'] == 'XGBRegressor':
        imp_model= xgb.XGBRegressor(**model_params['model_params'])
    else:
        pass

    imp_model = TransformedTargetRegressor(
        regressor=imp_model,
        inverse_func=lambda x: np.round(x),
        check_inverse=False,
    )

    # Make pipeline
    imp_pipeline = Pipeline(
        steps=[
            ('order_features', order_features),
            ('format_missing_data', format_missing_data),
            ('col_transformer', col_transformer),
            ('fill_nas', fill_nas),
            ('imp_model', imp_model),
        ]
    )

    # Add wrapper to skip model and return default for core features, and target values not satisfying the defined rules
    # default is set to median value of target
    default_response = np.nanmedian(imp_data_engred.loc[(imp_data_engred.group == 'train'), model_params['target']])
    filt_imp_pipeline = FilterPipeline(imp_pipeline, default_response)
    
    # StandEncoder can handle unseen categories : put 0 in all known terminal columns        
    # Only one-hot-encoded features can not support unseen cat. for now, add problematic encoders here :
    no_unknown_features = model_params['OneHotEncoder_features']
    missing_values = [np.nan, None, '']
    
    for feature_name in model_params['features_core']:
        excluded_values = missing_values
        if (feature_name in no_unknown_features) :
            feature_values = imp_data_engred.loc[(imp_data_engred.group == 'train')
                                                 & (imp_data_engred[feature_name].notnull()),
                                                 feature_name].unique().tolist()
            # Rules flagging unknown values (ignoring missing values)
            filt_imp_pipeline.add_include_rule(feature_name, feature_values + missing_values, 'Unknown ' + feature_name)
        if (feature_name in global_params['category_exclusions']) :
            excluded_values = missing_values + global_params['category_exclusions'][feature_name]
        # Rules flagging missing values and excluded
        filt_imp_pipeline.add_exclude_rule(feature_name, excluded_values, 'Missing/Excluded ' + feature_name)
     # Rules flagging invalid predictions/target
    filt_imp_pipeline.add_exclude_rule_preds(lambda x: x < 0, 'Negative prediction')

    # Train pipeline
    tic = time.time()
    cols = list(pd.unique(model_params['features'] + [model_params['target']] + ['group']))
    imp_data_engred = imp_data_engred[cols]
    imp_data_engred = imp_data_engred[imp_data_engred.notnull().all(axis=1)]


    filt_imp_pipeline.fit(
        imp_data_engred.loc[
            (imp_data_engred.group == 'train'),
            model_params['features']
        ],
        imp_data_engred.loc[
            (imp_data_engred.group == 'train'),
            'actual_arrival_ama_taxi_time'
        ],
    )
    toc = time.time()
    log = logging.getLogger(__name__)
    log.info('training impeded ama model took {:.1f} minutes'.format(
        (toc-tic)/60)
    )

    with mlflow.start_run(run_id=active_run_id):
        # Log trained model
        mlf_sklearn.log_model(
            sk_model=filt_imp_pipeline,
            artifact_path='model',
            conda_env=add_environment_specs_to_conda_file())
        # Set tags
        mlflow.set_tag('airport_icao', global_params['airport_icao'])
        mlflow.set_tag('Model Version', 0)
        # Log model parameters one at a time so that character limit is
        # 500 instead of 250
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        mlflow.log_params(global_params)

    return filt_imp_pipeline

def train_imp_ramp_model(
    aircraft_category: Dict[str,str],
    imp_data_engred: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,

) -> Pipeline:

    ac_enc=AircraftClassEncoder(
        aircraft_category,
    )

    stand_encoder = StandEncoder()
    
    AMA_ramp_taxi_enc = AvgRampTaxiEncoder()

     # Build one-hot encoder for general categorical features
    oh_enc=OneHotEncoder(
         sparse=False,
         handle_unknown="ignore",
     )

     # Make column transformer
    col_transformer=ColumnTransformer(
        [
            ('AMA_ramp_mean_taxi_time_encoder', AMA_ramp_taxi_enc, ['arrival_runway_actual','arrival_stand_actual']),
            ('aircraft_class_encoder', ac_enc, ['aircraft_type']),
            ('one_hot_encoder', oh_enc, model_params['OneHotEncoder_features']),
            ('stand_encoder', stand_encoder, ['arrival_stand_actual'] )
        ],
         remainder='passthrough',
     )

    # Replaces miscellaneous missing values with expected values
    format_missing_data = FormatMissingData(model_params['features_core'])

    # Replace NAs with provided value, only affecting numeric/bool since categorical are handled at the encoder level
    fill_nas = FillNAs(0)

    # Orders feature columns
    order_features = OrderFeatures()

    #  Make model
    if model_params['model'] == 'RandomForestRegressor':
        imp_model = RandomForestRegressor(**model_params['model_params'])
    elif model_params['model'] == 'XGBRegressor':
        imp_model= xgb.XGBRegressor(**model_params['model_params'])
    elif model_params['model'] == 'GradientBoostingRegressor':
        imp_model = GradientBoostingRegressor(**model_params['model_params'])
    else:
        pass

    imp_model = TransformedTargetRegressor(
        regressor=imp_model,
        inverse_func=lambda x: np.round(x),
        check_inverse=False,
    )

    # Make pipeline

    imp_pipeline = Pipeline(
        steps=[
            ('order_features', order_features),
            ('format_missing_data', format_missing_data),
            ('col_transformer', col_transformer),
            ('fill_nas', fill_nas),
            ('imp_model', imp_model),
        ]
    )

    # Add wrapper to skip model and return default for core features, and target values not satisfying the defined rules
    # default is set to median value of target
    default_response = np.nanmedian(imp_data_engred.loc[(imp_data_engred.group == 'train'), model_params['target']])
    filt_imp_pipeline = FilterPipeline(imp_pipeline, default_response)
    
    # StandEncoder can handle unseen categories : put 0 in all known terminal columns        
    # Only one-hot-encoded features can not support unseen cat. for now, add problematic encoders here :
    no_unknown_features = model_params['OneHotEncoder_features']
    missing_values = [np.nan, None, '']
    
    for feature_name in model_params['features_core']:
        excluded_values = missing_values
        if (feature_name in no_unknown_features) :
            feature_values = imp_data_engred.loc[(imp_data_engred.group == 'train')
                                                 & (imp_data_engred[feature_name].notnull()),
                                                 feature_name].unique().tolist()
            # Rules flagging unknown values (ignoring missing values)
            filt_imp_pipeline.add_include_rule(feature_name, feature_values + missing_values, 'Unknown ' + feature_name)
        if (feature_name in global_params['category_exclusions']) :
            excluded_values = missing_values + global_params['category_exclusions'][feature_name]
        # Rules flagging missing values and excluded
        filt_imp_pipeline.add_exclude_rule(feature_name, excluded_values, 'Missing/Excluded ' + feature_name)
     # Rules flagging invalid predictions/target
    filt_imp_pipeline.add_exclude_rule_preds(lambda x: x < 0, 'Negative prediction')

    # Train pipeline
    tic = time.time()
    cols = list(pd.unique(model_params['features'] + [model_params['target']] + ['group']))
    imp_data_engred = imp_data_engred[cols]
    imp_data_engred = imp_data_engred[imp_data_engred.notnull().all(axis=1)]

    filt_imp_pipeline.fit(
        imp_data_engred.loc[
            (imp_data_engred.group == 'train'),
            model_params['features']
        ],
        imp_data_engred.loc[
            (imp_data_engred.group == 'train'),
            model_params['target']
        ],
    )
    toc = time.time()
    log = logging.getLogger(__name__)
    log.info('training impeded ramp model took {:.1f} minutes'.format(
        (toc-tic)/60)
    )

    with mlflow.start_run(run_id=active_run_id):
        # Log trained model
        mlf_sklearn.log_model(
            sk_model=filt_imp_pipeline,
            artifact_path='model',
            conda_env=add_environment_specs_to_conda_file())
        # Set tags
        mlflow.set_tag('airport_icao', global_params['airport_icao'])
        mlflow.set_tag('Model Version', 0)
        # Log model parameters one at a time so that character limit is
        # 500 instead of 250
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        mlflow.log_params(global_params)

    return filt_imp_pipeline
