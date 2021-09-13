"""Pipeline for unimpeded taxi-in data query and save
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipelines(**kwargs):
    unimp_AMA_pipeline = Pipeline(
        [
            node(
                func=query_and_save_version_fraction_speed_gte_threshold,
                inputs=[
                    "fraction_speed_gte_threshold_data_set@DB",
                    "params:globals.airport_icao"
                ],
                outputs=None,
            ),
        ]
    )

    unimp_STBO_data_sql = Pipeline(
        [
            node(
                func= lambda x : [x],
                inputs=[
                    "ffs_data_set@DB"
                ],
                outputs=[
                    "ffs_data_set@CSV"
                ],
            ),
        ]
    )



    return {
        'unimp_ama': unimp_AMA_pipeline,
        'unimp_STBO_data_sql': unimp_STBO_data_sql
    }
