from kedro.pipeline import Pipeline, node

from ...data_engineering.nodes import *
from .nodes import *
from data_services.compute_surface_counts import compute_arrival_departure_count
from data_services.gate_occupied_at_landing_proxy import compute_gate_occupied_at_landing_proxy


def create_impeded_taxi_in_pipeline(**kwargs):
    
    imp_common_pipeline = Pipeline(
        [
            node(
                func=replace_runway_actuals,
                inputs=["MFS_data_set@CSV", "runway_actuals_data_set@CSV"],
                outputs="MFS_data_set_0",
            ),
            node(
                func=keep_second_of_duplicate_gufi,
                inputs="MFS_data_set_0",
                outputs="MFS_data_set_1",
            ),
            node(
                func=coalesce_actual_times,
                inputs="MFS_data_set_1",
                outputs="MFS_data_set_2",
            ),
            node(
                func=handle_round_robin_flights,
                inputs="MFS_data_set_2",
                outputs="data_0",
            ),
            node(
                func=prepare_aircraft_class_map,
                inputs="aircrafts_classes_map@CSV",
                outputs="aircraft_categories"
            )
        ]
    )

    imp_ramp_pipeline = Pipeline(
        [
            node(
                func=compute_arrival_departure_count,
                inputs="data_0",
                outputs="data_0_0"
            ),
            node(
                func=compute_gate_occupied_at_landing_proxy,
                inputs="data_0_0",
                outputs="data_0_1"
            ),
            node(
                func=add_arrival_stand_airline_time_at_landing,
                inputs=["data_0_1","airline_taxi_in_predictions_data_set@CSV"],
                outputs="data_0_2"
            ),
            node(
                func=add_scheduled_taxi_in_predictions,
                inputs=["data_0_2","scheduled_taxi_in_predictions_data_set@CSV"],
                outputs="data_0_3"
            ),
            node(
                func=compute_total_taxi_time,
                inputs="data_0_3",
                outputs="data_0_4"
            ),
            node(
                func=apply_filter_only_arrivals,
                inputs="data_0_4",
                outputs="data_0_5"
            ),
            node(
                func=add_train_test_group,
                inputs=["data_0_5", "parameters"],
                outputs="imp_data_engred"
            ),
            node(
                func=apply_imp_filters,
                inputs=["imp_data_engred", "params:imp_ramp_model_params"],
                outputs="imp_data_engred_filtered",
             )
        ]
    )

    imp_ama_pipeline = Pipeline(
        [
            node(
                func=compute_arrival_departure_count,
                inputs="data_0",
                outputs="data_0_0"
            ),
            node(
                func=compute_gate_occupied_at_landing_proxy,
                inputs="data_0_0",
                outputs="data_0_1"
            ),
            node(
                func=add_arrival_stand_airline_time_at_landing,
                inputs=["data_0_1","airline_taxi_in_predictions_data_set@CSV"],
                outputs="data_0_2"
            ),
            node(
                func=add_scheduled_taxi_in_predictions,
                inputs=["data_0_2","scheduled_taxi_in_predictions_data_set@CSV"],
                outputs="data_0_3"
            ),
            node(
                func=compute_total_taxi_time,
                inputs="data_0_3",
                outputs="data_0_4"
            ),
            node(
                func=apply_filter_only_arrivals,
                inputs="data_0_4",
                outputs="data_0_5"
            ),
            node(
                func=add_train_test_group,
                inputs=["data_0_5", "parameters"],
                outputs="imp_data_engred"
            ),
            node(
                func=apply_imp_filters,
                inputs=["imp_data_engred", "params:imp_ama_model_params"],
                outputs="imp_data_engred_filtered",
             )
        ]
    )
    
    return {
        'imp_ramp':
        imp_common_pipeline +
        imp_ramp_pipeline,
        'imp_ama':
        imp_common_pipeline +
        imp_ama_pipeline,
    }
