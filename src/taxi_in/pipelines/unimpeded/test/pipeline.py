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

    unimp_ramp_extra_nodes = Pipeline(
        [
            node(
                func=test_missing_features,
                inputs=[
                    "data_predicted",
                    "model_pipeline",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_unknown_features,
                inputs=[
                    "data_predicted",
                    "model_pipeline",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_category_exclusions,
                inputs=[
                    "data_predicted",
                    "model_pipeline",
                    "params:unimp_ramp_model_params",
                    "params:globals.category_exclusions",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_features_order,
                inputs=[
                    "data_predicted",
                    "model_pipeline",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_range,
                inputs=[
                    "data_predicted",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_type,
                inputs=[
                    "data_predicted",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_valid,
                inputs=[
                    "data_predicted",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_scale,
                inputs=[
                    "data_predicted",
                    "params:unimp_ramp_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
        ]
    )

    unimp_AMA_extra_nodes = Pipeline(
        [
            node(
                func=test_missing_features,
                inputs=[
                    "data_predicted",
                    "model_pipeline",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_unknown_features,
                inputs=[
                    "data_predicted",
                    "model_pipeline",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_category_exclusions,
                inputs=[
                    "data_predicted",
                    "model_pipeline",
                    "params:unimp_ama_model_params",
                    "params:globals.category_exclusions",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_features_order,
                inputs=[
                    "data_predicted",
                    "model_pipeline",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_range,
                inputs=[
                    "data_predicted",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_type,
                inputs=[
                    "data_predicted",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_valid,
                inputs=[
                    "data_predicted",
                    "params:unimp_ama_model_params",
                    "active_run_id",
                ],
                outputs=None,
            ),
            node(
                func=test_predicted_scale,
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
        'unimp_ramp': test_pipeline + unimp_ramp_extra_nodes,
        'unimp_ama': test_pipeline + unimp_AMA_extra_nodes,
    }