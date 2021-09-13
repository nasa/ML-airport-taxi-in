"""Nodes specific to unimpeded taxi in data query and save
"""

from typing import Any, Dict
from kedro.io import DataCatalog, Version
from kedro.extras.datasets.pandas import CSVDataSet

import pandas as pd
import numpy as np

from ...data_query_and_save.nodes import query_save_version

def query_and_save_version_fraction_speed_gte_threshold(
    data: pd.DataFrame,
    airport_icao: str
):
    query_save_version(
        data,
        airport_icao+".fraction_speed_gte_threshold",
        "01_raw",
    )




