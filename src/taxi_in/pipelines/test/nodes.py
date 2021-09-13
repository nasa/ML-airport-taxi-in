"""Testing nodes for unimpeded model development
"""

from typing import Any, Dict, Tuple
import logging

import pandas as pd
import numpy as np

import mlflow
from sklearn.pipeline import Pipeline as sklearn_Pipeline

import random
from copy import deepcopy
from math import isclose


def test_valid_prediction(
    model_pipeline: sklearn_Pipeline,
    data,
    model_params: Dict[str, Any],
    passed: int,
    tests: int,
    warning_string='',
) -> Tuple:

    # Predict and confirm get a valid output
    try:
        prediction = model_pipeline.predict(data)[0]
    except ValueError:
        prediction = False  # indicator that prediction failed

    if isinstance(prediction, target_type(model_params)):
        passed += 1
    else:
        log = logging.getLogger(__name__)
        log.warning(
            'failed to generate valid prediction ' +
            warning_string
        )

    tests += 1

    return (passed, tests)


def test_missing_features(
    data: pd.DataFrame,
    model_pipeline: sklearn_Pipeline,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Node for testing whether the model returns a valid value when
    missing features that should be imputed are passed in.
    """
    # Prep for metrics & logging
    passed = 0
    tests = 0
    log = logging.getLogger(__name__)

    # Grab a random entry from data
    random_idx = random_not_null_idx(data, model_params)

    log.info('Testing feature imputation')
    features_not_core = [v for v in model_params['features'] if v not in model_params['features_core']]

    for feature_impute in features_not_core:
        data_copy = deepcopy(data.loc[[random_idx], model_params['features']])

        # Remove features that should be imputed
        # using None
        data_copy.loc[random_idx, feature_impute] = None

        passed, tests = test_valid_prediction(
            model_pipeline,
            data_copy,
            model_params,
            passed,
            tests,
            'when ' + feature_impute + ' is None',
        )

        # Remove features that should be imputed
        # using np.nan
        data_copy.loc[random_idx, feature_impute] = np.nan

        passed, tests = test_valid_prediction(
            model_pipeline,
            data_copy,
            model_params,
            passed,
            tests,
            'when ' + feature_impute + ' is np.nan',
        )

        # Remove features that should be imputed
        # using empty string
        data_copy.loc[random_idx, feature_impute] = ''

        passed, tests = test_valid_prediction(
            model_pipeline,
            data_copy,
            model_params,
            passed,
            tests,
            'when ' + feature_impute + ' is empty string',
        )

    # Log results
    if tests > 0:
        with mlflow.start_run(run_id=active_run_id):
            mlflow.log_metric(
                'unit_test_fraction_feature_imputed',
                passed / tests,
            )


def test_unknown_features(
    data: pd.DataFrame,
    model_pipeline: sklearn_Pipeline,
    model_params: Dict[str, Any],
    active_run_id: str,
    test_same_as_missing=True,
) -> None:
    """Node for testing whether the model returns a valid value when
    unknown categories for features are passed in.
    """
    # Prep for metrics & logging
    passed = 0
    log = logging.getLogger(__name__)

    # Grab a random entry from data
    random_idx = random_not_null_idx(data, model_params)

    log.info('Testing unknown feature categories')

    features_str = [feature for feature in model_params['features']
                    if any([type(v) == str for v in data[feature].values])]

    for feature_handle_unknown in features_str:
        data_copy = deepcopy(data.loc[[random_idx], model_params['features']])

        # Make sure 'unknown_category' isn't actually a category
        assert 'unknown_category'\
            not in data[feature_handle_unknown].unique(),\
            'test failure because category called unknown_category'

        data_copy.loc[random_idx, feature_handle_unknown] = 'unknown_category'

        # Predict and confirm get a valid output
        try:
            prediction = model_pipeline.predict(data_copy)[0]
        except ValueError:
            prediction = False  # indicator that prediction failed

        if test_same_as_missing and (feature_handle_unknown not in model_params['features_core']):
            # Remove features that should be imputed
            data_copy.loc[random_idx, feature_handle_unknown] = None  # is this how to remove a feature?

            # Predict again
            prediction_missing = model_pipeline.predict(data_copy)[0]

            if isclose(
                prediction,
                prediction_missing,
                rel_tol=1e-2,
            ):
                passed += 1
            else:
                log.warning(
                    '\tunknown category for feature ' +
                    feature_handle_unknown +
                    ' produces different prediction than when missing'
                )
        else:
            if isinstance(prediction, target_type(model_params)):
                passed += 1
            else:
                log.warning(
                    '\tfailed handling unknown category for feature: ' +
                    feature_handle_unknown
                )

    # Log results
    if len(features_str) > 0:
        with mlflow.start_run(run_id=active_run_id):
            mlflow.log_metric(
                'unit_test_fraction_feature_handled_unknown',
                passed / len(features_str),
            )


def test_category_exclusions(
    data: pd.DataFrame,
    model_pipeline: sklearn_Pipeline,
    model_params: Dict[str, Any],
    category_exclusions: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Node for testing whether the model returns a valid value when an
    excluded feature category is provided.
    Also checks that the prediction is the same as when an unknown category
    is provided.
    """
    # Prep for metrics & logging
    passed = 0
    tests = 0
    log = logging.getLogger(__name__)

    # Grab a random entry from data
    random_idx = random_not_null_idx(data, model_params)

    log.info('Testing excluded feature categories')

    if isinstance(category_exclusions, dict):  # is list when empty
        for feature_w_exclusions in category_exclusions.keys():
            for excluded_category in category_exclusions[feature_w_exclusions]:
                excluded_category = str(excluded_category)  # ensure is a string

                data_copy = deepcopy(data.loc[[random_idx], model_params['features']])

                data_copy.loc[random_idx, feature_w_exclusions] = excluded_category

                # Predict and confirm get a valid output
                try:
                    prediction_excluded_cat = model_pipeline.predict(data_copy)[0]
                except ValueError:
                    prediction_excluded_cat = False  # indicator that prediction failed

                # Now make it an unknown category and see what get
                # Make sure 'unknown_category' isn't actually a category
                assert 'unknown_category'\
                    not in data[feature_w_exclusions].unique(),\
                    'test failure because category called unknown_category'

                data_copy.loc[random_idx, feature_w_exclusions] = 'unknown_category'

                # Predict with unknown category
                prediction_unknown_cat = model_pipeline.predict(data_copy)[0]

                if isclose(
                    prediction_excluded_cat,
                    prediction_unknown_cat,
                    rel_tol=1e-2,  # within 1%
                ) or (feature_w_exclusions in model_params['features_core'] and np.isnan(prediction_excluded_cat)):
                    passed += 1
                else:
                    log.warning(
                        'failed handling excluded category ' +
                        excluded_category +
                        ' for feature: ' +
                        feature_w_exclusions
                    )

                tests += 1

        # Log results
        if tests > 0:
            with mlflow.start_run(run_id=active_run_id):
                mlflow.log_metric(
                    'unit_test_fraction_feature_handled_exclusions',
                    passed / tests,
                )


def test_features_order(
    data: pd.DataFrame,
    model_pipeline: sklearn_Pipeline,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Test changing column order
    """
    # Grab a random entry from data
    passed=0
    random_idx = random_not_null_idx( data, model_params )

    data_copy = deepcopy(data.loc[[random_idx], model_params['features']+ ['predicted_{}'.format(model_params['name'])] ])
    prediction = data_copy['predicted_{}'.format(model_params['name'])].values[0]

    # Invert order
    data_copy = data_copy[data_copy.columns[::-1]]

    # Predict and confirm get a valid output
    try:
        prediction_inv_order = model_pipeline.predict(data_copy[model_params['features']])[0]
    except ValueError:
        prediction_inv_order = False  # indicator that prediction failed

    if prediction == prediction_inv_order:
        passed = 1

    # Log results
    with mlflow.start_run(run_id=active_run_id):
        mlflow.log_metric(
            'unit_test_features_order_change',
            passed,
        )


def test_predicted_range(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Test range of output values
    """
    predictions = data['predicted_{}'.format(model_params['name'])]

    if 'target_min' in model_params['unit_tests']:
        out_of_bound = predictions < model_params['unit_tests']['target_min']
    if 'target_max' in model_params['unit_tests']:
        out_of_bound = out_of_bound | (predictions < model_params['unit_tests']['target_max'])

    passed = int(sum( out_of_bound ) == 0)

    # Log results
    with mlflow.start_run(run_id=active_run_id):
        mlflow.log_metric(
            'unit_test_predicted_range',
            passed,
        )


def test_predicted_type(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Test predictions data type
    """
    passed = 0
    # Select single not null value
    random_idx = random_not_null_idx( data, model_params )
    prediction = data.loc[[random_idx], 'predicted_{}'.format(model_params['name'])].values[0]

    passed = int(isinstance(prediction, target_type(model_params)))

    # Log results
    with mlflow.start_run(run_id=active_run_id):
        mlflow.log_metric(
            'unit_test_predicted_type',
            passed,
        )


def test_predicted_valid(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Test predictions data type
    """

    predictions = data['predicted_{}'.format(model_params['name'])]

    percent_valid = (len(predictions) - sum(predictions.isnull())) / len(predictions)

    # ratio of not null predictions
    passed = int(
        percent_valid >= model_params['unit_tests']['target_min_valid_ratio']
    )

    # Log results
    with mlflow.start_run(run_id=active_run_id):
        mlflow.log_metric(
            'unit_test_predicted_valid',
            passed,
        )

    if not passed:
        log = logging.getLogger(__name__)
        log.warning(
            'failed test prediction valid unit test because ' +
            'only {:.1f}% were not null '.format(percent_valid*100) +
            'but unit test requires {:.1f}% or more to be not null'.format(
                model_params['unit_tests']['target_min_valid_ratio']*100
            )
        )
        log.warning(
            'of the null predictions, ' +
            '{:.1f}% were null because of missing core features'.format(
                (sum(predictions[data.missing_core_features].isnull()) /
                 sum(predictions.isnull()))*100
            )
        )


def test_predicted_scale(
    data: pd.DataFrame,
    model_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """Test predictions data scale (i.e. decimal places)
    """

    # Grab a random entry from data
    passed = 0
    random_idx = random_not_null_idx( data, model_params )
    prediction = data.loc[[random_idx], 'predicted_{}'.format(model_params['name'])].values[0]

    # After pushing target_scale decimals to the left no more decimals should be remaining.
    passed = int(((prediction * (10 ** model_params['unit_tests'][
        'target_scale'])) % 1) < 0.00000001 ) # comparing with small non-zero value due to floating point precision

    # Log results
    with mlflow.start_run(run_id=active_run_id):
        mlflow.log_metric(
            'unit_test_predicted_scale',
            passed,
        )


def target_type(model_params):
    types = ()
    if model_params['unit_tests']['target_type'] == 'float':
        types = (float, np.floating)
    else:
        if model_params['unit_tests']['target_type'] == 'int':
            types = (int, np.int)

    return types


def random_not_null_idx(data, model_params):

    not_null_idx = data['predicted_{}'.format(model_params['name'])].isnull() == False
    return random.choice(data[not_null_idx].index)
