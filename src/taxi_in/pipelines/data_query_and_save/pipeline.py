"""Pipeline for overall taxi-in data query and save
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=query_save_version_FFS,
                inputs=[
                    "MFS_data_set@DB",
                    "params:globals.airport_icao"
                ],
                outputs=None,
            ),
            node(
                func=query_and_save_version_runway_actuals,
                inputs=[
                    "runway_actuals_data_set@DB",
                    "params:globals.airport_icao"
                ],
                outputs=None,
            ),
        ]
    )
