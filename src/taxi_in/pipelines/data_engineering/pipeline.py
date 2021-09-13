"""General-purpose pipeline for data engineering for taxi-in prediction
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=replace_runway_actuals,
                inputs=["MFS_data_set@CSV", "runway_actuals_data_set@CSV"],
                outputs="data_init",
            ),
            node(
                func=merge_STBO,
                inputs=["data_init", "ffs_data_set@CSV"],
                outputs="data_0"
            ),
            node(
                func=keep_second_of_duplicate_gufi,
                inputs="data_0",
                outputs="data_1",
            ),
            node(
                func=coalesce_actual_times,
                inputs="data_1",
                outputs="data_2"
            ),
            node(
                func=handle_round_robin_flights,
                inputs="data_2",
                outputs="data_3",
            ),
            node(
                func=set_index,
                inputs="data_3",
                outputs="data_4"
            ),
            node(
                func=compute_total_taxi_time,
                inputs="data_4",
                outputs="data_5"
            ),
            node(
                func=add_train_test_group,
                inputs=["data_5", "parameters"],
                outputs="data_engred_general",
            ),
        ]
    )


