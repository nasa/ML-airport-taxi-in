"""Pipelines for unimpeded taxi-in prediction test
"""
from kedro.pipeline import Pipeline, node

from taxi_in.pipelines import test
from ...test.nodes import test_missing_features
from ...test.nodes import test_unknown_features
from ...test.nodes import test_category_exclusions
from ...test.nodes import test_features_order
from ...test.nodes import test_predicted_range
from ...test.nodes import test_predicted_type
from ...test.nodes import test_predicted_valid
from ...test.nodes import test_predicted_scale

def create_pipelines(**kwargs):
    test_pipeline = test.create_pipeline()

    imp_AMA_extra_nodes = Pipeline(
        [
            node(
                func=test_missing_features,
                inputs=[
                    "imp_data_predicted",
                    "imp_model_pipeline",
                    "params:imp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_unknown_features,
                inputs=[
                    "imp_data_predicted",
                    "imp_model_pipeline",
                    "params:imp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_category_exclusions,
                inputs=[
                    "imp_data_predicted",
                    "imp_model_pipeline",
                    "params:imp_ama_model_params",
                    "params:globals.category_exclusions",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_features_order,
                inputs=[
                    "imp_data_predicted",
                    "imp_model_pipeline",
                    "params:imp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_range,
                inputs=[
                    "imp_data_predicted",
                    "params:imp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_type,
                inputs=[
                    "imp_data_predicted",
                    "params:imp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_valid,
                inputs=[
                    "imp_data_predicted",
                    "params:imp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_scale,
                inputs=[
                    "imp_data_predicted",
                    "params:imp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
        ]
    )

    imp_ramp_extra_nodes = Pipeline(
        [
            node(
                func=test_missing_features,
                inputs=[
                    "imp_data_predicted",
                    "imp_model_pipeline",
                    "params:imp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_unknown_features,
                inputs=[
                    "imp_data_predicted",
                    "imp_model_pipeline",
                    "params:imp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_category_exclusions,
                inputs=[
                    "imp_data_predicted",
                    "imp_model_pipeline",
                    "params:imp_ramp_model_params",
                    "params:globals.category_exclusions",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_features_order,
                inputs=[
                    "imp_data_predicted",
                    "imp_model_pipeline",
                    "params:imp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_range,
                inputs=[
                    "imp_data_predicted",
                    "params:imp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_type,
                inputs=[
                    "imp_data_predicted",
                    "params:imp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_valid,
                inputs=[
                    "imp_data_predicted",
                    "params:imp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_scale,
                inputs=[
                    "imp_data_predicted",
                    "params:imp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
        ]
    )

    return {
        'imp_ramp': test_pipeline + imp_ramp_extra_nodes,
        'imp_ama': test_pipeline + imp_AMA_extra_nodes,
    }