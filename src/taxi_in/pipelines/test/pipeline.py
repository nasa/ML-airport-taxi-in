"""General-purpose pipeline for model testing for taxi-in prediction
(testing for things like handling of missing features)
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        []
    )
