"""Pipeline for unimpeded taxi-in prediction
"""
from kedro.pipeline import Pipeline, node

from ...data_science.nodes import *
from .nodes import *
from data_services.mlflow_utils import init_mlflow, init_mlflow_run
from data_services.network import copy_artifacts_to_ntx


def create_pipelines(**kwargs):
    unimp_ramp_pipeline = Pipeline(
        [
            node(
                func=init_mlflow,
                inputs="params:unimp_ramp_model_params",
                outputs="experiment_id",
            ),
            node(
                func=init_mlflow_run,
                inputs=[
                    "params:unimp_ramp_model_params",
                    "experiment_id",
                ],
                outputs="active_run_id",
            ),
            node(
                func=train_unimp_ramp_model,
                inputs=[
                    "data_engred_unimp_ramp",
                    "params:unimp_ramp_model_params",
                    "params:globals",
                    "active_run_id",
                ],
                outputs="model_pipeline",
            ),
            node(
                func=train_baseline,
                inputs=[
                    "data_engred_unimp_ramp",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs="baseline_pipeline",
            ),
            node(
                func=predict,
                inputs=[
                    "model_pipeline",
                    "data_engred_unimp_ramp",
                    "params:unimp_ramp_model_params",
                ],
                outputs="data_predicted_m",
            ),
            node(
                func=predict_baseline,
                inputs=[
                    "baseline_pipeline",
                    "data_predicted_m",
                    "params:unimp_ramp_model_params",
                ],
                outputs="data_predicted",
            ),
            node(
                func=report_performance_metrics,
                inputs=[
                    "data_predicted",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs="artifacts_ready",
            ),
            node(
                func=log_model_sample_data,
                inputs=[
                    'data_engred_unimp_ramp',
                    'params:unimp_ramp_model_params',
                    'active_run_id',
                ],
                outputs=None,
                name='log_models_sample_data',
            ),
            node(
                func=copy_artifacts_to_ntx,
                inputs=[
                    "experiment_id",
                    "active_run_id",
                    "params:ntx_connection",
                    "artifacts_ready",
                ],
                outputs=None,
            )
        ]
    )

    unimp_ramp_validation_pipeline = Pipeline(
        [
            node(
                func=init_mlflow,
                inputs="params:unimp_ramp_model_params",
                outputs="experiment_id",
            ),
            node(
                func=init_mlflow_run,
                inputs=[
                    "params:unimp_ramp_model_params",
                    "experiment_id",
                    "validate",
                ],
                outputs="active_run_id",
            ),
            node(
                func=load_unimp_ramp_mlflow_model,
                inputs=[
                    "params:globals",
                    "active_run_id",
                ],
                outputs="model_pipeline",
            ),
            node(
                func=load_unimp_ramp_mlflow_baseline,
                inputs=[
                    "params:globals",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs="baseline_pipeline",
            ),
            node(
                func=predict,
                inputs=[
                    "model_pipeline",
                    "data_engred_unimp_ramp",
                    "params:unimp_ramp_model_params",
                ],
                outputs="data_predicted_m",
            ),
            node(
                func=predict_baseline,
                inputs=[
                    "baseline_pipeline",
                    "data_predicted_m",
                    "params:unimp_ramp_model_params",
                ],
                outputs="data_predicted",
            ),
            node(
                func=report_performance_metrics,
                inputs=[
                    "data_predicted",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            )
        ]
    )

    unimp_ama_pipeline = Pipeline(
        [
            node(
                func=init_mlflow,
                inputs="params:unimp_ama_model_params",
                outputs="experiment_id",
            ),
            node(
                func=init_mlflow_run,
                inputs=[
                    "params:unimp_ama_model_params",
                    "experiment_id",
                ],
                outputs="active_run_id",
            ),
            node(
                func=train_unimp_ama_model,
                inputs=[
                    "data_engred_unimp_ama",
                    "params:unimp_ama_model_params",
                    "params:globals",
                    "active_run_id",
                ],
                outputs="model_pipeline",
            ),
            node(
                func=train_baseline,
                inputs=[
                    "data_engred_unimp_ama",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs="baseline_pipeline",
            ),
            node(
                func=predict,
                inputs=[
                    "model_pipeline",
                    "data_engred_unimp_ama",
                    "params:unimp_ama_model_params",
                ],
                outputs="data_predicted_m",
            ),
            node(
                func=predict_baseline,
                inputs=[
                    "baseline_pipeline",
                    "data_predicted_m",
                    "params:unimp_ama_model_params",
                ],
                outputs="data_predicted",
            ),
            node(
                func=report_performance_metrics_ama,
                inputs=[
                    "data_predicted",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs="artifacts_ready",
            ),
            node(
                func=log_model_sample_data,
                inputs=[
                    'data_engred_unimp_ama',
                    'params:unimp_ama_model_params',
                    'active_run_id',
                ],
                outputs=None,
                name='log_models_sample_data',
            ),
            node(
                func=copy_artifacts_to_ntx,
                inputs=[
                    "experiment_id",
                    "active_run_id",
                    "params:ntx_connection",
                    "artifacts_ready",
                ],
                outputs=None,
            )
        ]
    )

    unimp_ama_validation_pipeline = Pipeline(
        [
            node(
                func=init_mlflow,
                inputs="params:unimp_ama_model_params",
                outputs="experiment_id",
            ),
            node(
                func=init_mlflow_run,
                inputs=[
                    "params:unimp_ama_model_params",
                    "experiment_id",
                    "validate",
                ],
                outputs="active_run_id",
            ),
            node(
                func=load_unimp_ama_mlflow_model,
                inputs=[
                    "params:globals",
                    "active_run_id",
                ],
                outputs="model_pipeline",
            ),
            node(
                func=load_unimp_ama_mlflow_baseline,
                inputs=[
                    "params:globals",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs="baseline_pipeline",
            ),
            node(
                func=predict,
                inputs=[
                    "model_pipeline",
                    "data_engred_unimp_ama",
                    "params:unimp_ama_model_params",
                ],
                outputs="data_predicted_m",
            ),
            node(
                func=predict_baseline,
                inputs=[
                    "baseline_pipeline",
                    "data_predicted_m",
                    "params:unimp_ama_model_params",
                ],
                outputs="data_predicted",
            ),
            node(
                func=report_performance_metrics_ama,
                inputs=[
                    "data_predicted",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
        ]
    )

    return {
        'unimp_ramp': unimp_ramp_pipeline,
        'unimp_ramp_validation': unimp_ramp_validation_pipeline,
        'unimp_ama': unimp_ama_pipeline,
        'unimp_ama_validation': unimp_ama_validation_pipeline,
    }
