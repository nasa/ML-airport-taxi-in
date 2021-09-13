"""Construction of the master pipeline.
"""

from typing import Dict
from kedro.pipeline import Pipeline

from taxi_in.pipelines import data_query_and_save as dqs
from taxi_in.pipelines import data_engineering as de
from taxi_in.pipelines import data_science as ds
from taxi_in.pipelines import test

from taxi_in.pipelines.unimpeded import data_query_and_save as unimp_dqs
from taxi_in.pipelines.unimpeded import data_engineering as unimp_de
from taxi_in.pipelines.unimpeded import data_science as unimp_ds
from taxi_in.pipelines.unimpeded import test as unimp_test

from taxi_in.pipelines.impeded import data_query_and_save as imp_dqs
from taxi_in.pipelines.impeded import data_engineering as imp_de
from taxi_in.pipelines.impeded import data_science as imp_ds
from taxi_in.pipelines.impeded import test as imp_test

def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    data_query_and_save_pipeline = dqs.create_pipeline()
    data_engineering_pipeline = de.create_pipeline()
    test_pipeline = test.create_pipeline()

    unimp_dqs_pipelines = unimp_dqs.create_pipelines()
    unimp_de_pipelines = unimp_de.create_pipelines()
    unimp_ds_pipelines = unimp_ds.create_pipelines()
    unimp_test_pipelines = unimp_test.create_pipelines()

    imp_taxi_in_dqs_pipeline = imp_dqs.create_pipeline()
    imp_taxi_in_de_pipeline = imp_de.create_impeded_taxi_in_pipeline()
    imp_taxi_in_ds_pipeline = imp_ds.create_pipelines()
    imp_taxi_in_test_pipeline = imp_test.create_pipelines()

    return {
        "dqs": data_query_and_save_pipeline,
        "de": data_engineering_pipeline,
        "unimp_ramp_de": unimp_de_pipelines['unimp_ramp'],
        "unimp_ramp_ds": unimp_ds_pipelines['unimp_ramp'],
        "unimp_ramp_test": unimp_test_pipelines['unimp_ramp'],
        "unimp_ramp_full": unimp_de_pipelines['unimp_ramp'] + unimp_ds_pipelines['unimp_ramp'] + unimp_test_pipelines['unimp_ramp'],
        "unimp_ramp_validate": unimp_de_pipelines['unimp_ramp'] + unimp_ds_pipelines['unimp_ramp_validation'],
        "unimp_dqs": unimp_dqs_pipelines['unimp_ama'],
        "unimp_dqs_STBO": unimp_dqs_pipelines['unimp_ama'] + unimp_dqs_pipelines['unimp_STBO_data_sql'],
        "unimp_ama_de": unimp_de_pipelines['unimp_ama'],
        "unimp_ama_full": unimp_de_pipelines['unimp_ama'] + unimp_ds_pipelines['unimp_ama'] + unimp_test_pipelines['unimp_ama'],
        "unimp_ama_validate": unimp_de_pipelines['unimp_ama'] + unimp_ds_pipelines['unimp_ama_validation'],
        "imp_dqs": imp_taxi_in_dqs_pipeline,
        "imp_ama_de": imp_taxi_in_de_pipeline['imp_ama'],
        "imp_ama_ds": imp_taxi_in_ds_pipeline['imp_ama'],
        "imp_ama_full": imp_taxi_in_de_pipeline['imp_ama'] + imp_taxi_in_ds_pipeline['imp_ama'] + imp_taxi_in_test_pipeline['imp_ama'],
        "imp_ama_validate": imp_taxi_in_de_pipeline['imp_ama']+ imp_taxi_in_ds_pipeline['imp_ama_validation'],
        "imp_ramp_de": imp_taxi_in_de_pipeline['imp_ramp'],
        "imp_ramp_ds": imp_taxi_in_ds_pipeline['imp_ramp'],
        "imp_ramp_full": imp_taxi_in_de_pipeline['imp_ramp'] + imp_taxi_in_ds_pipeline['imp_ramp'] + imp_taxi_in_test_pipeline['imp_ramp'],
        "imp_ramp_validate": imp_taxi_in_de_pipeline['imp_ramp'] + imp_taxi_in_ds_pipeline['imp_ramp_validation'],
        "__default__": unimp_de_pipelines['unimp_ramp'] + unimp_ds_pipelines['unimp_ramp'] + unimp_test_pipelines['unimp_ramp'],
    }
