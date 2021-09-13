"""General-purpose data science nodes
"""

import os
import logging
from typing import Any, Dict, List
import time
import pickle

import pandas as pd
import numpy as np

import mlflow
from mlflow import sklearn as mlf_sklearn

from sklearn.pipeline import Pipeline as sklearn_Pipeline

from data_services.mlflow_utils import (get_best_model, 
                          get_most_recent_registered_model, 
                          get_model_by_run_id, 
                          add_environment_specs_to_conda_file)

from .evaluation_utils import evaluate_predictions
from .error_metrics import METRIC_NAME_TO_FUNCTION_DICT
from .baseline import GroupByModel


def predict(
    pipeline: sklearn_Pipeline,
    data: pd.DataFrame,
    model_params: Dict[str, Any],
) -> pd.DataFrame:
    # Run model
    tic = time.time()
    predictions = pipeline.predict(
        data[model_params['features']]
    )
    toc = time.time()

    log = logging.getLogger(__name__)
    log.info('predicting took {:.1f} minutes'.format(
        (toc-tic)/60)
    )

    # Add predictions to dataframe for convenience
    data['predicted_{}'.format(model_params['name'])] = predictions
    # Flag predictions with missing core features
    data['missing_core_features'] = pipeline._filter(data[model_params['features']]) == False

    return data


def report_performance_metrics(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Node for reporting performance metrics. Notice that this function has no
    outputs.
    """

    # Predictive model
    report_model_metrics(data,
                         model_params,
                         'predicted_{}'.format(model_params['name']),
                         active_run_id)
    # Baseline
    if 'predicted_baseline' in data.columns:
        report_model_metrics(data, 
                             model_params, 
                             'predicted_baseline',
                             active_run_id,
                             'baseline_',
                             ['test'])

    # STBO as truth data
    if ('label' in model_params) :
      if 'undelayed_arrival_{}_transit_time'.format(model_params['label']) in data.columns:
        report_model_metrics_STBO(data, 
                                  model_params, 
                                  'predicted_{}'.format(model_params['name']),
                                  active_run_id,
                                  'STBO_',
                                  ['test'])

        
    return True


def report_model_metrics(
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
    """

    metrics_dict = {
        metric_name: METRIC_NAME_TO_FUNCTION_DICT[metric_name]
        for metric_name in model_params['metrics']
    }

    evaluation_df = evaluate_predictions(
        data[(data.missing_core_features == False) &
              (data['predicted_{}'.format(model_params['name'])].isna() == False) &
               data.group.isin(group_values)],
        y_true=model_params['target'],
        y_pred=y_pred,
        metrics_dict=metrics_dict,
    )

    # Log the accuracy of the model
    log = logging.getLogger(__name__)

    with mlflow.start_run(run_id=active_run_id):
        # Set the metrics
        for metric_name in metrics_dict.keys():
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

def report_model_metrics_STBO(
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
    """

    metrics_dict = {
        metric_name: METRIC_NAME_TO_FUNCTION_DICT[metric_name]
        for metric_name in model_params['metrics']
    }

    if 'undelayed_arrival_{}_transit_time'.format(model_params['label']) in data.columns:
        data_filtered = data[(data.missing_core_features == False) &
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


def train_baseline(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:

    if 'baseline' in model_params:
        baseline_model = GroupByModel(model_params['baseline'])
        baseline_model.fit(data[data.group == 'train'],
                           data[data.group == 'train'][model_params['target']])

        with mlflow.start_run(run_id=active_run_id):
            # Log trained model
            mlf_sklearn.log_model(
                sk_model=baseline_model,
                artifact_path='baseline',
                conda_env=add_environment_specs_to_conda_file())

    else:
        baseline_model = []

    return baseline_model


def predict_baseline(
    pipeline: sklearn_Pipeline,
    data: pd.DataFrame,
    model_params: Dict[str, Any]
) -> pd.DataFrame:

    if 'baseline' in model_params:
        data['predicted_baseline'] = pipeline.predict(data)

    return data


def load_mlflow_model(
    globals: Dict[str, Any],
    model_name: str,
    active_run_id: str,
    artifact_name: str='model/model.pkl',
) -> sklearn_Pipeline:

    if model_name in globals['validate']:
        if 'run_id' in globals['validate'][model_name]:
            run_id = globals['validate'][model_name]['run_id']

            model_path = get_model_by_run_id(
                run_id,
                artifact_name
            )

        else:
            raise ValueError(
                'invalid model to validate: {}'.format(globals['validate']['model_name'])
            )
    else:
        if globals['validate']['default'] == 'get_most_recent_registered_model':
            run_id, model_path = get_most_recent_registered_model(
                name=model_name + '_' + globals['airport_icao'],
            )
        elif globals['validate']['default'] == 'get_best_model':
            run_id, model_path = get_best_model(
                name=model_name + '_' + globals['airport_icao'],
            )
        else:
            raise ValueError(
                'invalid model to validate: {}'.format(globals['validate']['default'])
            )

    with open(model_path, 'rb') as f:
        model_pipeline = pickle.load(f)

    with mlflow.start_run(run_id=active_run_id):
        # MLflow logging
        mlflow.log_params(globals)
        mlflow.set_tag('airport_icao', globals['airport_icao'])
        mlflow.set_tag('Test', 'validate')
        mlflow.set_tag('validate_run_id', run_id)
        

    return model_pipeline


def load_unimp_ramp_mlflow_model(
    globals: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:
        
    return load_mlflow_model(
        globals=globals,
        model_name='unimpeded_taxi_in_ramp',
        active_run_id=active_run_id,
    )


def load_unimp_ramp_mlflow_baseline(
    globals: Dict[str, Any],
    model_params: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:

    if 'baseline' in model_params:
        return load_mlflow_model(
            globals=globals,
            model_name='unimpeded_taxi_in_ramp',
            active_run_id=active_run_id,
            artifact_name='baseline/model.pkl'
        )
    else:
        return []


def load_unimp_ama_mlflow_model(
    globals: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:
    return load_mlflow_model(
        globals=globals,
        model_name='unimpeded_taxi_in_AMA',
        active_run_id=active_run_id,
    )


def load_unimp_ama_mlflow_baseline(
    globals: Dict[str, Any],
    model_params: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:

    if 'baseline' in model_params:
        return load_mlflow_model(
            globals=globals,
            model_name='unimpeded_taxi_in_AMA',
            active_run_id=active_run_id,
            artifact_name='baseline/model.pkl'
        )
    else:
        return []


def load_imp_ramp_mlflow_model(
    globals: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:
    return load_mlflow_model(
        globals=globals,
        model_name='impeded_taxi_in_ramp',
        active_run_id=active_run_id,
    )


def load_imp_ramp_mlflow_baseline(
    globals: Dict[str, Any],
    model_params: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:

    if 'baseline' in model_params:
        return load_mlflow_model(
            globals=globals,
            model_name='impeded_taxi_in_ramp',
            active_run_id=active_run_id,
            artifact_name='baseline/model.pkl'
        )
    else:
        return []


def load_imp_ama_mlflow_model(
    globals: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:
    return load_mlflow_model(
        globals=globals,
        model_name='impeded_taxi_in_AMA',
        active_run_id=active_run_id,
    )


def load_imp_ama_mlflow_baseline(
    globals: Dict[str, Any],
    model_params: Dict[str, Any],
    active_run_id: str,
) -> sklearn_Pipeline:

    if 'baseline' in model_params:
        return load_mlflow_model(
            globals=globals,
            model_name='impeded_taxi_in_AMA',
            active_run_id=active_run_id,
            artifact_name='baseline/model.pkl'
        )
    else:
        return []


def log_model_sample_data(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
    data_set_size: int = 10,
    groups: List[str] = ['train', 'test'],
):
    log = logging.getLogger(__name__)

    # Create sample data
    input_data = data.loc[
        data.group.isin(groups),
        model_params['features']
    ]
    sampled_index = np.random.choice(
        input_data.index,
        size=data_set_size,
        replace=False,
    )
    input_data = input_data.loc[sampled_index, :]
    target_data = data.loc[
        sampled_index,
        model_params['target']
    ]

    input_data_path_name = os.path.join(
        'data',
        '05_model_input',
        'sample_input_data.csv',
    )
    target_data_path_name = os.path.join(
        'data',
        '05_model_input',
        'sample_target_data.csv',
    )

    input_data.to_csv(input_data_path_name)
    target_data.to_csv(target_data_path_name)

    # Restart the MLflow run we used when training this model
    with mlflow.start_run(
        run_id=active_run_id
    ) as active_run:

        log.info('logging sample data for model {}'.format(
            model_params['name']
        ))

        mlflow.log_artifact(input_data_path_name, 'model')
        mlflow.log_artifact(target_data_path_name, 'model')

    os.remove(input_data_path_name)
    os.remove(target_data_path_name)
