from typing import Dict

from kedro.pipeline import Pipeline, node

from ...data_engineering.nodes import *
from .nodes import *

from taxi_in.pipelines import data_engineering as de


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    data_engineering_pipeline = de.create_pipeline()

    unimp_extra_nodes = Pipeline(
        [
            node(
                func=apply_filter_only_arrivals,
                inputs="data_engred_general",
                outputs="data_engred_general_arrivals",
            ),
            node(
                func=apply_filter_null_times,
                inputs="data_engred_general_arrivals",
                outputs="data_engred_general_arrivals_non_null"),
            node(
                func=apply_filter_req_arr_stand_and_runway,
                inputs="data_engred_general_arrivals_non_null",
                outputs="data_engred_general_arrivals_filtered",
            ),
        ]
    )

    unimp_ramp_extra_nodes = Pipeline(
        [
            node(
                func=apply_filter_req_arr_ramp_taxi_times,
                inputs="data_engred_general_arrivals_filtered",
                outputs="data_engred_unimp_ramp",
            ),
            node(
                func=train_test_group_logging,
                inputs="data_engred_unimp_ramp",
                outputs=None,
            ),
        ]
    )

    unimp_ama_extra_nodes = Pipeline(
        [
            node(
                func=apply_filter_req_arr_ama_taxi_times,
                inputs="data_engred_general_arrivals_filtered",
                outputs="data_engred_general_arrivals_filtered_ama",
            ),
            node(
                func=set_index,
                inputs="fraction_speed_gte_threshold_data_set@CSV",
                outputs="fraction_speed_gte_threshold",
            ),
            node(
                func=join_fraction_speed_gte_threshold_and_filter,
                inputs=[
                    "data_engred_general_arrivals_filtered_ama",
                    "fraction_speed_gte_threshold"
                ],
                outputs="data_joined_unimp_ama",
            ),
            node(
                func=calculate_unimpeded_AMA,
                inputs=[
                    "data_joined_unimp_ama",
                    "params:unimp_ama_model_params",
                ],
                outputs="data_engred_unimp_ama",
            ),
            node(
                func=train_test_group_logging,
                inputs="data_engred_unimp_ama",
                outputs=None,
            ),
        ]
    )

    return {
        'unimp_ramp':
        data_engineering_pipeline +
        unimp_extra_nodes +
        unimp_ramp_extra_nodes,
        'unimp_ama':
        data_engineering_pipeline +
        unimp_extra_nodes +
        unimp_ama_extra_nodes,
    }
