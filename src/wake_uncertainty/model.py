import os
import sys
import time
import joblib
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .config import FEATURES, TARGET, RANDOM_SEED
from .exception import CustomException
from .logger import logging


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "best_model.pkl")


def prepare_data(df, features=FEATURES, target=TARGET):
    """
    Prepare the dataset for XGBoost model training or evaluation.

    The function checks that all required feature and target columns are
    present in the input DataFrame, removes rows where the target value is
    missing, and separates the data into input features (X) and target
    values (y).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the model features and target column.
    features : list of str, optional
        Names of the columns used as model inputs. Defaults to FEATURES.
    target : str, optional
        Name of the target column to predict. Defaults to TARGET.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame containing only the selected model features.
    y : pandas.Series
        Series containing the target values.

    Raises
    ------
    CustomException
        If an unexpected error occurs while preparing the data.

    Example
    -------
    >>> X, y = prepare_data(training_df)
    >>> print(X.shape)
    >>> print(y.shape)
    """
    try:
        missing_columns = [column for column in list(features) + [target]
                           if column not in df.columns]

        if missing_columns:
            logging.info(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        clean = df.dropna(subset=[target]).copy()

        X = clean[list(features)]
        y = clean[target]

        logging.info(f"Prepared {len(X)} rows with {len(features)} features.")

        return X, y

    except Exception as e:
        logging.exception(f"Failed to prepare data: {e}")
        raise CustomException(e, sys)


def build_model(random_seed=RANDOM_SEED, **kwargs):
    """
    Build the XGBoost regression pipeline used by the surrogate model.

    The pipeline consists of a median-value imputer followed by an
    XGBoost regression model. The imputer allows the model to handle
    missing feature values without requiring the input dataset to be
    completely complete.

    Parameters
    ----------
    random_seed : int, optional
        Random seed used by XGBoost to make model training reproducible.
        Defaults to RANDOM_SEED.
    **kwargs
        Additional XGBoost hyperparameters that override the default
        model configuration, such as n_estimators, max_depth, or
        learning_rate.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A scikit-learn Pipeline containing a median SimpleImputer and
        an XGBRegressor.

    Raises
    ------
    CustomException
        If the model pipeline cannot be created.

    Example
    -------
    >>> model = build_model(n_estimators=300, max_depth=6)
    >>> model.fit(X_train, y_train)
    """
    try:
        params = {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": random_seed,
            "n_jobs": -1,
        }

        params.update(kwargs)

        model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("model", XGBRegressor(**params)),])

        return model

    except Exception as e:
        logging.exception(f"Failed to build XGBoost model: {e}")
        raise CustomException(e, sys)


def evaluate_model(model, X, y):
    """
    Evaluate a trained regression model using MAE, RMSE, and R².

    Parameters
    ----------
    model : sklearn estimator
        A trained regression model with a ``predict`` method.
    X : pandas.DataFrame
        Input feature data used for evaluation.
    y : pandas.Series or numpy.ndarray
        Observed target values corresponding to X.

    Returns
    -------
    dict
        Dictionary containing:
        - ``mae`` : Mean absolute error.
        - ``rmse`` : Root mean squared error.
        - ``r2`` : Coefficient of determination.

    Raises
    ------
    CustomException
        If model evaluation fails.

    Example
    -------
    >>> metrics = evaluate_model(model, X_test, y_test)
    >>> print(metrics["rmse"])
    >>> print(metrics["r2"])
    """
    try:
        y_pred = model.predict(X)

        metrics = {"mae": float(mean_absolute_error(y, y_pred)),
                   "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                   "r2": float(r2_score(y, y_pred))}

        logging.info(f"Model evaluation: {metrics}")

        return metrics

    except Exception as e:
        logging.exception(f"Model evaluation failed: {e}")
        raise CustomException(e, sys)


def train_model(df, features=FEATURES, target=TARGET, test_size=0.2, random_seed=RANDOM_SEED, **kwargs):
    """
    Train and evaluate the XGBoost surrogate model.

    The function prepares the input dataset, splits it into training and
    testing subsets, builds the XGBoost pipeline, trains the model, and
    evaluates it on the held-out test data. Training time and evaluation
    metrics are stored together with the trained model.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the model features and target variable.
    features : list of str, optional
        Names of the input feature columns. Defaults to FEATURES.
    target : str, optional
        Name of the target variable. Defaults to TARGET.
    test_size : float, optional
        Fraction of the dataset reserved for testing. Defaults to 0.2,
        meaning 20% of the data is used for testing.
    random_seed : int, optional
        Random seed used for the train/test split and model training.
        Defaults to RANDOM_SEED.
    **kwargs
        Additional XGBoost hyperparameters passed to ``build_model``.

    Returns
    -------
    dict
        Model bundle containing:
        - ``model`` : Trained XGBoost pipeline.
        - ``features`` : Feature names used by the model.
        - ``target`` : Target variable name.
        - ``model_type`` : Model type, ``"xgboost"``.
        - ``metrics`` : Test-set MAE, RMSE, R² and training time.
        - ``random_seed`` : Random seed used during training.

    Raises
    ------
    CustomException
        If training or evaluation fails.

    Example
    -------
    >>> bundle = train_model(
    ...     training_df,
    ...     test_size=0.2,
    ...     n_estimators=500,
    ...     max_depth=8
    ... )
    >>> print(bundle["metrics"])
    """
    try:
        X, y = prepare_data(df, features=features, target=target)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        model = build_model(random_seed=random_seed, **kwargs)

        start = time.perf_counter()

        model.fit(X_train, y_train)

        train_seconds = time.perf_counter() - start

        metrics = evaluate_model(model, X_test, y_test)

        metrics["train_seconds"] = float(train_seconds)

        bundle = {
            "model": model,
            "features": list(features),
            "target": target,
            "model_type": "xgboost",
            "metrics": metrics,
            "random_seed": random_seed,
        }

        logging.info(f"XGBoost training complete. "
                     f"R2={metrics['r2']:.4f}, "
                     f"RMSE={metrics['rmse']:.6f}")

        return bundle

    except Exception as e:
        logging.exception(f"XGBoost training failed: {e}")
        raise CustomException(e, sys)


def predict(frame, model=None):
    """
    Generate predictions using a trained surrogate model.

    The function loads the default saved model when no model is supplied.
    It then checks that all required model features are present in the
    input DataFrame and generates predictions for each row.

    Parameters
    ----------
    frame : pandas.DataFrame
        Input DataFrame containing the features required by the trained
        model.
    model : sklearn estimator or dict, optional
        Trained model or model bundle returned by ``train_model``.
        If omitted, the default saved model is loaded automatically.

    Returns
    -------
    numpy.ndarray
        Array containing one predicted target value for each input row.

    Raises
    ------
    CustomException
        If prediction fails.

    Example
    -------
    >>> predictions = predict(test_df, model=bundle)
    >>> print(predictions[:5])
    """
    try:
        if model is None:
            model = load_model()

        if isinstance(model, dict):
            estimator = model["model"]
            features = model.get("features", FEATURES)
        else:
            estimator = model
            features = FEATURES

        missing = [feature for feature in features if feature not in frame.columns]

        if missing:
            raise ValueError(f"Missing prediction features: {missing}")

        prediction = estimator.predict(frame[features])

        return np.asarray(prediction)

    except Exception as e:
        logging.exception(f"Prediction failed: {e}")
        raise CustomException(e, sys)


def save_model(bundle, path=None):
    """
    Save a trained model bundle to disk using joblib.

    The function creates the destination directory if it does not already
    exist and saves the supplied model bundle as a joblib file.

    Parameters
    ----------
    bundle : dict
        Trained model bundle returned by train_model.
    path : str or None, optional
        Destination path for the saved model. If omitted, the default
        model path defined by DEFAULT_MODEL_PATH is used.

    Returns
    -------
    str
        Path to the saved model file.

    Raises
    ------
    CustomException
        If the model cannot be saved.

    Example
    -------
    >>> bundle = train_model(training_df)
    >>> model_path = save_model(bundle)
    >>> print(model_path)
    """
    try:
        model_path = path if path is not None else DEFAULT_MODEL_PATH

        model_dir = os.path.dirname(model_path)

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        joblib.dump(bundle, model_path)

        logging.info(f"Model saved to {model_path}")

        return model_path

    except Exception as e:
        logging.exception(f"Failed to save model: {e}")
        raise CustomException(e, sys)


def load_model(path=None):
    """
    Load a previously trained surrogate model from disk.

    The function loads the model bundle saved using save_model.
    If no path is supplied, it loads the default model from
    DEFAULT_MODEL_PATH.

    Parameters
    ----------
    path : str or None, optional
        Path to the saved joblib model file. If omitted, the default
        model path is used.

    Returns
    -------
    dict or sklearn estimator
        The saved trained model or model bundle.

    Raises
    ------
    CustomException
        If the model cannot be loaded.

    Example
    -------
    >>> model = load_model()
    >>> predictions = predict(test_df, model=model)
    """
    try:
        model_path = path if path is not None else DEFAULT_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        logging.info(f"Model loaded from {model_path}")

        return model

    except Exception as e:
        logging.exception(f"Failed to load model: {e}")
        raise CustomException(e, sys)
