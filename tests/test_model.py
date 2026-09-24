import numpy as np
import pandas as pd

from wake_uncertainty.config import FEATURES, TARGET
from wake_uncertainty.model import prepare_data, build_model, train_model, predict


def make_test_dataframe(n=100):
    """
    Create a synthetic DataFrame for testing the machine-learning pipeline.

    The generated data contains the same input features used by the
    wake-uncertainty model. The target variable is generated from a simple
    deterministic relationship between wind speed, blocking count and
    blocking ratio.

    Parameters
    ----------
    n : int, optional
        Number of synthetic turbine observations to generate.
        Default is 100.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the configured model features and the
        ``power_ratio`` target variable.

    Example
    -------
    Create a test dataset containing 200 observations:

    df = make_test_dataframe(n=200)
    """
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "ws_free": rng.uniform(4, 20, n),
        "wd": rng.uniform(0, 360, n),
        "ti": rng.uniform(0.04, 0.10, n),
        "target_rotor_dia_m": np.full(n, 164.0),
        "target_hub_height_m": np.full(n, 105.0),
        "target_rated_power_kw": np.full(n, 8400.0),
        "nb_distance_km": rng.uniform(5, 50, n),
        "nb_direction_deg": rng.uniform(0, 360, n),
        "nb_n_turbines": rng.integers(20, 100, n),
        "nb_spacing_D": rng.choice([3, 5, 7, 9], n),
        "nb_rotor_dia_m": np.full(n, 178.0),
        "nb_hub_height_m": np.full(n, 119.0),
        "nb_rated_power_kw": np.full(n, 10000.0),
        "pi_n_blocking": rng.integers(0, 10, n),
        "pi_blocking_ratio": rng.uniform(0, 1, n),
        "pi_blocking_distance": rng.uniform(0, 20000, n),
        "pi_streamwise_dist_1st": rng.uniform(0, 15000, n),
        "pi_lateral_dist_1st": rng.uniform(0, 1000, n),
        "pi_rotor_dia_1st": rng.choice([0, 164, 178], n),
    })

    df[TARGET] = 0.03 * df["ws_free"] - 0.002 * df["pi_n_blocking"] - 0.02 * df["pi_blocking_ratio"] + 0.4
    return df


def test_prepare_data_shapes():
    """
    Test that prepare_data returns feature and target arrays with
    the correct number of observations.

    Parameters
    ----------
    None
        This test does not take any parameters.

    Returns
    -------
    None

    Example
    -------
    Run the test case using the command line:

    pytest tests/test_model.py::test_prepare_data_shapes
    """
    df = make_test_dataframe()

    X, y = prepare_data(df)

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert list(X.columns) == FEATURES


def test_prepare_data_keeps_missing_features():
    """
    Test that prepare_data preserves missing feature values.

    This verifies that missing values are passed through the data-preparation
    stage and can subsequently be handled by the model pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    Run this test with:

    pytest tests/test_model.py::test_prepare_data_keeps_missing_features
    """
    df = make_test_dataframe()

    df.loc[0, "pi_blocking_distance"] = np.nan

    X, y = prepare_data(df)

    assert len(X) == len(df)
    assert X["pi_blocking_distance"].isna().sum() == 1


def test_xgboost_pipeline_contains_imputer():
    """
    Test that the XGBoost pipeline contains both an imputer and a model.

    The imputer is required to handle missing feature values before the
    XGBoost model receives the data.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    Run this test with:

    pytest tests/test_model.py::test_xgboost_pipeline_contains_imputer
    """
    model = build_model(n_estimators=5, max_depth=2)

    assert "imputer" in model.named_steps
    assert "model" in model.named_steps


def test_model_handles_missing_values():
    """
    Test that the XGBoost training pipeline can train when input features
    contain missing values.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    Run this test with:

    pytest tests/test_model.py::test_model_handles_missing_values
    """
    df = make_test_dataframe()
    df.loc[0:10, "pi_blocking_distance"] = np.nan
    bundle = train_model(df, test_size=0.2, n_estimators=5, max_depth=2)

    assert bundle["model"] is not None


def test_train_model_returns_xgboost_bundle():
    """
    Test that train_model returns the expected XGBoost model bundle.

    The test verifies the model type, feature list and prediction target stored
    in the returned bundle.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    Run this test with:

    pytest tests/test_model.py::test_train_model_returns_xgboost_bundle
    """
    df = make_test_dataframe()
    bundle = train_model(df, n_estimators=5, max_depth=2)

    assert bundle["model_type"] == "xgboost"
    assert bundle["features"] == FEATURES
    assert bundle["target"] == TARGET


def test_predict_returns_correct_length():
    """
    Test that the prediction function returns one prediction for each
    input observation.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    Run this test with:

    pytest tests/test_model.py::test_predict_returns_correct_length
    """
    df = make_test_dataframe()
    bundle = train_model(df, n_estimators=5, max_depth=2)
    predictions = predict(df, model=bundle)

    assert len(predictions) == len(df)


def test_predictions_are_finite():
    """
    Test that the trained model produces finite predictions.

    This verifies that the prediction pipeline does not produce NaN or
    infinite values for the synthetic test dataset.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example
    -------
    Run this test with:

    pytest tests/test_model.py::test_predictions_are_finite
    """
    df = make_test_dataframe()
    bundle = train_model(df, n_estimators=5, max_depth=2)
    predictions = predict(df, model=bundle)

    assert np.all(np.isfinite(predictions))
