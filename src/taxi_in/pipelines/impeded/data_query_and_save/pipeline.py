"""Pipeline for impeded taxi-in data query and save
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=query_save_version_airline_taxi_in_predictions,
                inputs=[
                    "airline_taxi_in_predictions_data_set@DB",
                    "params:globals.airport_icao"
                ],
                outputs=None,
            ),
            node(
                func=query_save_version_scheduled_taxi_in_predictions,
                inputs=[
                    "scheduled_taxi_in_predictions_data_set@DB",
                    "params:globals.airport_icao"
                ],
                outputs=None,
            ),
        ]
    )
