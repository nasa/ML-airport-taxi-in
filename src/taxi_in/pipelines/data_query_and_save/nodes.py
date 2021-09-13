"""Nodes for querying and saving data sets.
"""

from typing import Any, Dict
from kedro.io import DataCatalog, Version
from kedro.extras.datasets.pandas import CSVDataSet


import pandas as pd
import numpy as np


def query_save_version(
    data: pd.DataFrame,
    name: str,
    folder='01_raw',
    versioned=False,
):
    """Saves results of DB query to an @CSV version of the data set

    Note: Assumed that data comes from {name}_data_set@DB and then
    save resulting CSV to data/{folder}/{name}_data_set@CSV
    """
    if versioned:
        version = Version(
            load=None,
            save=None,
        )
    else:
        version = None

    data_set_CSV = CSVDataSet(
        filepath="data/{}/{}_data_set.csv".format(
            folder,
            name
        ),
        save_args={"index": False},
        version=version,
    )
    dc = DataCatalog({"{}_data_set@CSV".format(name): data_set_CSV})

    dc.save("{}_data_set@CSV".format(name), data)


def query_save_version_FFS(
    data: pd.DataFrame,
    airport_icao: str
):
    query_save_version(
        data,
        airport_icao+".MFS",
    )

def query_and_save_version_runway_actuals(
    data: pd.DataFrame,
    airport_icao: str
):
    query_save_version(
        data,
        airport_icao+".runway_actuals",
    )