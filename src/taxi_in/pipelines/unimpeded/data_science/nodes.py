"""Data science nodes for unimpeded model development
"""

from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from data_services.mlflow_utils import add_environment_specs_to_conda_file

import time
import logging

import mlflow
from mlflow import sklearn as mlf_sklearn

from data_services.stand_cluster_encoder import StandEncoder
from ...data_science.error_metrics import METRIC_NAME_TO_FUNCTION_DICT
from ...data_science.evaluation_utils import evaluate_predictions
from data_services.FilterPipeline import FilterPipeline
from data_services.OrderFeatures import OrderFeatures


def train_unimp_ramp_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:

    features_transforms = list()

    for feature in model_params['features']:
        feature_Pipeline_steps = list()

        if feature not in model_params['features_core']:
            # Impute to fill in nan
            impute_nan_w_None = SimpleImputer(
                missing_values=np.nan,
                strategy='constant',
                fill_value=None,
            )
            feature_Pipeline_steps.append((
                'impute_nan_w_None',
                impute_nan_w_None
            ))
            # Impute to fill in empty strings ('')
            impute_empty_string_w_None = SimpleImputer(
                missing_values='',
                strategy='constant',
                fill_value=None,
            )
            feature_Pipeline_steps.append((
                'impute_empty_string_w_None',
                impute_empty_string_w_None
            ))

        if feature in model_params['OneHotEncoder_features']:
            one_hot_enc = OneHotEncoder(
                sparse=False,
                handle_unknown="ignore",
            )
            feature_Pipeline_steps.append((
                'one_hot_enc',
                one_hot_enc
            ))

        if feature == 'arrival_stand_actual':
            stand_encoder = StandEncoder()
            feature_Pipeline_steps.append((
                'stand_encoder',
                stand_encoder
            ))

        feature_Pipeline = sklearn_Pipeline(feature_Pipeline_steps)
        feature_transforms = (feature, feature_Pipeline, [feature])
        features_transforms.append(feature_transforms)

    col_transformer = ColumnTransformer(
        transformers=features_transforms,
        remainder='passthrough',
        sparse_threshold=0,
    )

    # Orders feature columns
    order_features = OrderFeatures()

    # Make model
    if model_params['model'] == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(**model_params['model_params'])
    else:
        pass

    # Add rounding of model result to nearest integer
    # by somewhat mis-using a transformed target regressor
    # inputs should be integer number of seconds already
    model = TransformedTargetRegressor(
        regressor=model,
        inverse_func=lambda x: np.round(x),
        check_inverse=False,
    )

    # Make pipeline
    pipeline = sklearn_Pipeline(
        steps=[
            ('order_features', order_features),
            ('col_transformer', col_transformer),
            ('model', model),
        ]
    )

    # Add wrapper to skip model and return default for core features, and target values not satisfying the defined rules
    # default is set to the quantile seek by the model or the median of the value if no quantile requested
    if ((model_params['model'] == 'GradientBoostingRegressor') and
        ('loss' in model_params['model_params']) and
        (model_params['model_params']['loss'] == 'quantile')) :
        default_response = np.nanquantile(data.loc[(data.group == 'train'), model_params['target']],
                                          model_params['model_params']['alpha'])
    else :
        default_response = np.nanmedian(data.loc[(data.group == 'train'), model_params['target']])
    filt_pipeline = FilterPipeline(pipeline, default_response, print_debug=True)

    missing_values = [np.nan, None, '']
    # StandEncoder can handle unseen categories : put 0 in all known terminal columns
    # Only one-hot-encoded features can not support unseen cat. for now, add problematic encoders here :
    no_unknown_features =  model_params['OneHotEncoder_features']

    for feature_name in model_params['features_core']:
        excluded_values = missing_values
        if (feature_name in no_unknown_features) :
            feature_values = data.loc[(data.group == 'train') & (data[feature_name].notnull()), feature_name].unique().tolist()
            # Rules flagging unknown values (ignoring missing values)
            filt_pipeline.add_include_rule(feature_name, feature_values + missing_values, 'Unknown ' + feature_name)
        if (feature_name in global_params['category_exclusions']) :
            excluded_values = missing_values + global_params['category_exclusions'][feature_name]
        # Rules flagging missing values and excluded
        filt_pipeline.add_exclude_rule(feature_name, excluded_values, 'Missing/Excluded ' + feature_name)
    # Rules flagging invalid predictions/target
    filt_pipeline.add_exclude_rule_preds(lambda x: x < 0, 'Negative prediction')

    # Train pipeline
    tic = time.time()
    filt_pipeline.fit(
        data.loc[
            (data.group == 'train'),
            model_params['features']
        ],
        data.loc[
            (data.group == 'train'),
            model_params['target']
        ],
    )
    toc = time.time()

    log = logging.getLogger(__name__)
    log.info('training unimpeded ramp model took {:.1f} minutes'.format(
        (toc - tic) / 60)
    )
    
    with mlflow.start_run(run_id=active_run_id):
    
        # Log trained model
        mlf_sklearn.log_model(
            sk_model=filt_pipeline,
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

    return filt_pipeline


def train_unimp_ama_model(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:

    features_transforms = list()

    for feature in model_params['features']:
        feature_Pipeline_steps = list()

        if feature not in model_params['features_core']:
            # Impute to fill in nan
            impute_nan_w_None = SimpleImputer(
                missing_values=np.nan,
                strategy='constant',
                fill_value=None,
            )
            feature_Pipeline_steps.append((
                'impute_nan_w_None',
                impute_nan_w_None
            ))
            # Impute to fill in empty strings ('')
            impute_empty_string_w_None = SimpleImputer(
                missing_values='',
                strategy='constant',
                fill_value=None,
            )
            feature_Pipeline_steps.append((
                'impute_empty_string_w_None',
                impute_empty_string_w_None
            ))

        if feature in model_params['OneHotEncoder_features']:
            one_hot_enc = OneHotEncoder(
                sparse=False,
                handle_unknown='ignore',
            )
            feature_Pipeline_steps.append((
                'one_hot_enc',
                one_hot_enc
            ))

        if feature == 'arrival_stand_actual':
            stand_encoder = StandEncoder()
            feature_Pipeline_steps.append((
                'stand_encoder',
                stand_encoder
            ))

        feature_Pipeline = sklearn_Pipeline(feature_Pipeline_steps)
        feature_transforms = (feature, feature_Pipeline, [feature])
        features_transforms.append(feature_transforms)

    col_transformer = ColumnTransformer(
        transformers=features_transforms,
        remainder='passthrough',
        sparse_threshold=0,
    )

    # Orders feature columns
    order_features = OrderFeatures()

    if model_params['model'] == 'XGBRegressor':
        model = xgb.XGBRegressor(**model_params['model_params'])
    else:
        pass

    # Add rounding of model result to nearest integer
    # by somewhat mis-using a transformed target regressor
    # inputs should be integer number of seconds already
    model = TransformedTargetRegressor(
        regressor=model,
        inverse_func=lambda x: np.round(x),
        check_inverse=False,
    )

    # Make pipeline
    pipeline = sklearn_Pipeline(
        steps=[
            ('order_features', order_features),
            ('col_transformer', col_transformer),
            ('model', model),
        ]
    )

    # Add wrapper to skip model and return default for core features, and target values not satisfying the defined rules
    # default is set to the median of the target value
    default_response = np.nanmedian(data.loc[(data.unimpeded_AMA) & (data.group == 'train'), model_params['target']])
    filt_pipeline = FilterPipeline(pipeline, default_response, print_debug=True)
    missing_values = [np.nan, None,'']
    # StandEncoder can handle unseen categories : put 0 in all known terminal columns
    # Only one-hot-encoded features can not support unseen cat. for now, add problematic encoders here :
    no_unknown_features =  model_params['OneHotEncoder_features']

    for feature_name in model_params['features_core']:
        excluded_values = missing_values
        if (feature_name in no_unknown_features) :
            feature_values = data.loc[(data.group == 'train') & (data[feature_name].notnull()), feature_name].unique().tolist()
            # Rules flagging unknown values (ignoring missing values)
            filt_pipeline.add_include_rule(feature_name, feature_values + missing_values, 'Unknown ' + feature_name)
        if (feature_name in global_params['category_exclusions']) :
            excluded_values = missing_values + global_params['category_exclusions'][feature_name]
        # Rules flagging missing values and excluded
        filt_pipeline.add_exclude_rule(feature_name, excluded_values, 'Missing/Excluded ' + feature_name)
    # Rules flagging invalid predictions/target
    filt_pipeline.add_exclude_rule_preds(lambda x: x < 0, 'Negative prediction')

    
    # Train pipeline
    tic = time.time()
    filt_pipeline.fit(
        data.loc[
            (data.unimpeded_AMA) & (data.group == 'train'),
            model_params['features']
        ],
        data.loc[
            (data.unimpeded_AMA) & (data.group == 'train'),
            model_params['target']
        ],
    )
    toc = time.time()

    log = logging.getLogger(__name__)
    log.info('training unimpeded AMA model with {} unimpeded trajectories took {:.1f} minutes'.format(
        ((data.unimpeded_AMA) & (data.group == 'train')).sum(),
        (toc - tic) / 60)
    )
        
    with mlflow.start_run(run_id=active_run_id):

        # Log trained model
        mlf_sklearn.log_model(
            sk_model=filt_pipeline,
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

    return filt_pipeline


def report_performance_metrics_ama(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> bool:
    """Node for reporting performance metrics. Notice that this function has no
    outputs.
    """

    # Predictive model
    report_model_metrics_ama(data,
                             model_params,
                             'predicted_{}'.format(model_params['name']),
                             active_run_id)
    # Baseline
    if 'predicted_baseline' in data.columns:
        report_model_metrics_ama(data, 
                                 model_params, 
                                 'predicted_baseline',
                                 active_run_id,
                                 'baseline_',
                                 ['test'])
    
    # STBO as truth data
    if ('label' in model_params) :
      if 'undelayed_arrival_{}_transit_time'.format(model_params['label']) in data.columns:
        report_model_metrics_ama_STBO(data, 
                                  model_params, 
                                  'predicted_{}'.format(model_params['name']),
                                  active_run_id,
                                  'STBO_',
                                  ['test'])
        
    return True
        

def report_model_metrics_ama(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    y_pred: str,
    active_run_id: str,
    name_prefix: str = '',
    group_values: list = ['train', 'test']
) -> None:
    """Node for reporting the performance metrics of the predictions performed
    by the previous node. Notice that this function has no outputs, except
    logging.

    Primarily evaluates with the unimpeded_AMA trajectories, except for
    'fraction_less_than_actual', which uses all trajectories
    """

    metrics_dict = {
        metric_name: METRIC_NAME_TO_FUNCTION_DICT[metric_name]
        for metric_name in model_params['metrics']
        if metric_name != 'fraction_less_than_actual'
    }

    evaluation_df = evaluate_predictions(
        data[data.unimpeded_AMA & (data.missing_core_features == False) &
                    (data['predicted_{}'.format(model_params['name'])].isna() == False) &
                    data.group.isin(group_values)],
        y_true=model_params['target'],
        y_pred=y_pred,
        metrics_dict=metrics_dict,
    )

    if 'fraction_less_than_actual' in model_params['metrics']:
        evaluation_df_lt_actual = evaluate_predictions(
            data,
            y_true=model_params['target'],
            y_pred=y_pred,
            metrics_dict={
                'fraction_less_than_actual': METRIC_NAME_TO_FUNCTION_DICT['fraction_less_than_actual']
            },
        )
        evaluation_df = pd.concat(
            [evaluation_df, evaluation_df_lt_actual],
            axis=1,
        )

    # Log the accuracy of the model
    log = logging.getLogger(__name__)

    # Set active ML Flow Run
    with mlflow.start_run(run_id=active_run_id):

        for metric_name in model_params['metrics']:
            log.info("metric {}:".format(name_prefix + metric_name))
            for group in [v for v in data.group.unique() if v in group_values]:
                log.info("{} group: {}".format(
                    group,
                    evaluation_df.loc[group, metric_name]
                ))
                mlflow.log_metric(
                    name_prefix + metric_name + '_' + group,
                    evaluation_df.loc[group, metric_name]
                )


def report_model_metrics_ama_STBO(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    y_pred: str,
    active_run_id: str,
    name_prefix: str = '',
    group_values : list = ['train','test']
) -> None:
    """Node for reporting the performance metrics of the predictions performed
    by the previous node. Notice that this function has no outputs, except
    logging.

    Primarily evaluates with the unimpeded_AMA trajectories, except for
    'fraction_less_than_actual', which uses all trajectories
    """

    metrics_dict = {
        metric_name: METRIC_NAME_TO_FUNCTION_DICT[metric_name]
        for metric_name in model_params['metrics']
        if metric_name != 'fraction_less_than_actual'
    }

    if 'undelayed_arrival_{}_transit_time'.format(model_params['label']) in data.columns:
        data_filtered = data[(data.unimpeded_AMA) & (data.missing_core_features == False) &
              (data['predicted_{}'.format(model_params['name'])].isna() == False) &
              (data['predicted_baseline'].isna() == False) &
              (data['undelayed_arrival_{}_transit_time'.format(model_params['label'])].isna() == False) &
               data.group.isin(group_values)]
        evaluation_df_STBO = evaluate_predictions(
            data_filtered,
            y_true='undelayed_arrival_{}_transit_time'.format(model_params['label']),
            y_pred=y_pred,
            metrics_dict=metrics_dict,
        )

    if 'fraction_less_than_actual' in model_params['metrics']:
        evaluation_df_lt_actual = evaluate_predictions(
            data,
            y_true='undelayed_arrival_{}_transit_time'.format(model_params['label']),
            y_pred=y_pred,
            metrics_dict={
                'fraction_less_than_actual': METRIC_NAME_TO_FUNCTION_DICT['fraction_less_than_actual']
            },
        )
        evaluation_df_STBO = pd.concat(
            [evaluation_df_STBO, evaluation_df_lt_actual],
            axis=1,
        )


    # Log the accuracy of the model
    log = logging.getLogger(__name__)

    if (active_run_id != None) :
        with mlflow.start_run(run_id=active_run_id):
            # Set the metrics
            # for metric_name in metrics_dict.keys():
            # use evaluation_df to get the unimpeded_AMA metrics as well (if calculated)
            if 'undelayed_arrival_{}_transit_time'.format(model_params['label']) in data.columns:
                for metric_name in evaluation_df_STBO.keys() :
                    log.info("metric {}:".format(name_prefix + metric_name))
                    for group in [v for v in data.group.unique() if v in group_values]:
                        log.info("{} group: {}".format(
                            group,
                            evaluation_df_STBO.loc[group, metric_name]
                        ))
                        mlflow.log_metric(
                            name_prefix + metric_name + '_' + group,
                            evaluation_df_STBO.loc[group, metric_name]
                        )
