"""Nodes specific to impeded taxi in data query and save
"""

from typing import Any, Dict
from kedro.io import DataCatalog, Version
from kedro.extras.datasets.pandas import CSVDataSet

import pandas as pd
import numpy as np

from ...data_query_and_save.nodes import query_save_version

def query_save_version_airline_taxi_in_predictions(
    data: pd.DataFrame,
    airport_icao: str
):
    query_save_version(
        data,
        airport_icao+".airline_taxi_in_predictions",
        "03_primary",
    )


def query_save_version_scheduled_taxi_in_predictions(
    data: pd.DataFrame,
    airport_icao: str
):
    query_save_version(
        data,
        airport_icao+".scheduled_taxi_in_predictions",
        "03_primary",
    )