import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .data_labels import validate_label

TARGET_COLUMN = "global_active_energy_kwh"
HISTORICAL_AVERAGE = "historical_average"
LINEAR_REGRESSION = "linear_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"
MODEL_NAMES = (
    HISTORICAL_AVERAGE,
    LINEAR_REGRESSION,
    RANDOM_FOREST,
    XGBOOST,
)
LAG_HOURS = (24, 48, 72, 168)
ORIGIN_LAG_HOURS = (0, 1, 3, 6, 12)


@dataclass(frozen=True)
class ForecastConfig:
    target_column: str = TARGET_COLUMN
    label: str = "real"
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    alpha: float = 0.10
    random_state: int = 42

    def __post_init__(self):
        validate_label(self.label)
        fractions = self.train_fraction + self.validation_fraction
        if self.train_fraction <= 0 or self.validation_fraction <= 0 or fractions >= 1:
            raise ValueError("train and validation fractions must be positive and sum below 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex


@dataclass(frozen=True)
class ForecastResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict


def load_cleaned_energy(path, config=None):
    config = config if config is not None else ForecastConfig()
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if "dataset_label" not in frame.columns:
        raise ValueError("cleaned file must contain dataset_label")
    if config.target_column not in frame.columns:
        raise ValueError(f"cleaned file must contain {config.target_column}")
    labels = set(frame["dataset_label"].astype(str).str.lower().unique())
    expected_label = validate_label(config.label)
    if labels != {expected_label}:
        raise ValueError(f"expected dataset label {expected_label!r}, found {labels}")
    series = frame.set_index("timestamp")[config.target_column].sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("timestamp column must parse to a DatetimeIndex")
    if not series.index.is_monotonic_increasing:
        raise ValueError("energy index must be chronologically ordered")
    series.name = config.target_column
    return series


def build_day_ahead_features(energy, target=None, fill_missing=True, exogenous=None):
    energy = energy.sort_index()
    history = energy.ffill() if fill_missing else energy
    target_series = target.sort_index() if target is not None else energy
    target_index = target_series.index[target_series.notna()]
    origin_index = target_index.normalize() - pd.Timedelta(hours=1)

    features = pd.DataFrame(index=target_index)
    features["hour"] = target_index.hour
    features["dayofweek"] = target_index.dayofweek
    features["dayofyear"] = target_index.dayofyear
    features["month"] = target_index.month
    features["day"] = target_index.day
    features["weekend"] = (target_index.dayofweek >= 5).astype(int)
    features["hour_sin"] = np.sin(2.0 * np.pi * target_index.hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * target_index.hour / 24.0)
    features["year_sin"] = np.sin(2.0 * np.pi * target_index.dayofyear / 365.25)
    features["year_cos"] = np.cos(2.0 * np.pi * target_index.dayofyear / 365.25)

    for lag in LAG_HOURS:
        lookup = target_index - pd.Timedelta(hours=lag)
        features[f"lag_{lag}h"] = history.reindex(lookup).to_numpy(dtype=float)
    for lag in ORIGIN_LAG_HOURS:
        lookup = origin_index - pd.Timedelta(hours=lag)
        features[f"origin_lag_{lag}h"] = history.reindex(lookup).to_numpy(dtype=float)

    rolling_mean = history.rolling("24h", min_periods=24).mean()
    rolling_std = history.rolling("24h", min_periods=24).std()
    features["origin_roll_mean_24h"] = rolling_mean.reindex(origin_index).to_numpy(dtype=float)
    features["origin_roll_std_24h"] = rolling_std.reindex(origin_index).to_numpy(dtype=float)

    if exogenous is not None:
        if not isinstance(exogenous, pd.DataFrame):
            raise ValueError("exogenous must be a pandas DataFrame")
        if not exogenous.index.is_unique:
            raise ValueError("exogenous index must be unique")
        for column in exogenous.columns:
            features[f"exog_{column}"] = exogenous[column].reindex(target_index).to_numpy(dtype=float)

    target_values = target_series.reindex(target_index).astype(float)
    target_values.name = "actual_kwh"
    complete = features.notna().all(axis=1) & target_values.notna()
    return features.loc[complete], target_values.loc[complete]


def chronological_split(index, config=None):
    config = config if config is not None else ForecastConfig()
    index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing:
        raise ValueError("index must be chronologically ordered")

    unique_days = pd.DatetimeIndex(index.normalize().unique())
    train_days = max(1, int(len(unique_days) * config.train_fraction))
    validation_days = max(1, int(len(unique_days) * config.validation_fraction))
    if train_days + validation_days >= len(unique_days):
        raise ValueError("not enough distinct days for the requested split")

    train_end = unique_days[train_days - 1]
    validation_start = train_end + pd.Timedelta(days=1)
    validation_end = unique_days[train_days + validation_days - 1]
    test_start = validation_end + pd.Timedelta(days=1)

    normalized = index.normalize()
    train_mask = normalized <= train_end
    validation_mask = (normalized >= validation_start) & (normalized <= validation_end)
    test_mask = normalized >= test_start

    train_index = index[train_mask]
    validation_index = index[validation_mask]
    test_index = index[test_mask]
    if len(train_index) == 0 or len(validation_index) == 0 or len(test_index) == 0:
        raise ValueError("each chronological split must contain rows")
    if train_index.max() >= validation_index.min():
        raise ValueError("train and validation overlaps")
    if validation_index.max() >= test_index.min():
        raise ValueError("validation and test overlaps")
    return TimeSplit(train=train_index, validation=validation_index, test=test_index)


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    mape = np.nan if np.any(actual <= 0) else float(100.0 * np.mean(np.abs(error) / actual))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": mape,
    }


def historical_average_predict(train_features, train_target, eval_features):
    training = pd.DataFrame(
        {
            "hour": train_features["hour"].to_numpy(),
            "dayofweek": train_features["dayofweek"].to_numpy(),
            "target": np.asarray(train_target, dtype=float),
        }
    )
    by_hour_weekday = training.groupby(["hour", "dayofweek"])["target"].mean()
    by_hour = training.groupby("hour")["target"].mean()
    overall_mean = float(training["target"].mean())

    eval_hour = eval_features["hour"].to_numpy()
    eval_day = eval_features["dayofweek"].to_numpy()
    keys = pd.MultiIndex.from_arrays([eval_hour, eval_day])
    predicted = by_hour_weekday.reindex(keys).to_numpy(dtype=float)

    missing = pd.isna(predicted)
    if missing.any():
        predicted[missing] = by_hour.reindex(eval_hour[missing]).to_numpy(dtype=float)
    missing = pd.isna(predicted)
    if missing.any():
        predicted[missing] = overall_mean
    return predicted


def build_models(config=None):
    config = config if config is not None else ForecastConfig()
    return {
        LINEAR_REGRESSION: make_pipeline(StandardScaler(), LinearRegression()),
        RANDOM_FOREST: RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=config.random_state,
            n_jobs=-1,
        ),
        XGBOOST: XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def conformal_radius(actual, predicted, alpha):
    residuals = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    n = len(residuals)
    if n == 0:
        raise ValueError("calibration set must not be empty")
    rank = min(max(math.ceil((n + 1) * (1.0 - alpha)), 1), n)
    return float(np.partition(residuals, rank - 1)[rank - 1])


def _split_metadata(index, split):
    def bounds(values):
        return {
            "start": values.min().isoformat(),
            "end": values.max().isoformat(),
            "rows": int(len(values)),
        }

    return {
        "supervised_rows": int(len(index)),
        "train": bounds(split.train),
        "validation": bounds(split.validation),
        "test": bounds(split.test),
    }


def run_forecast_comparison(cleaned_path, config=None):
    config = config if config is not None else ForecastConfig()
    energy = load_cleaned_energy(cleaned_path, config)
    features, target = build_day_ahead_features(energy)
    split = chronological_split(features.index, config)

    features_train = features.loc[split.train]
    features_validation = features.loc[split.validation]
    features_test = features.loc[split.test]
    target_train = target.loc[split.train]
    target_validation = target.loc[split.validation]
    target_test = target.loc[split.test]

    predictions = pd.DataFrame(index=features.index)
    predictions["actual_kwh"] = target.to_numpy(dtype=float)
    predictions[HISTORICAL_AVERAGE] = np.nan
    predictions.loc[split.train, HISTORICAL_AVERAGE] = historical_average_predict(
        features_train, target_train, features_train
    )
    predictions.loc[split.validation, HISTORICAL_AVERAGE] = historical_average_predict(
        features_train, target_train, features_validation
    )
    predictions.loc[split.test, HISTORICAL_AVERAGE] = historical_average_predict(
        features_train, target_train, features_test
    )

    for name, model in build_models(config).items():
        model.fit(features_train, target_train)
        predictions.loc[split.validation, name] = model.predict(features_validation)
        predictions.loc[split.test, name] = model.predict(features_test)

    validation_rows = []
    for name in MODEL_NAMES:
        metrics = regression_metrics(
            target_validation, predictions.loc[split.validation, name]
        )
        validation_rows.append(
            {"model": name, "mae": metrics["mae"], "rmse": metrics["rmse"]}
        )
    validation_frame = pd.DataFrame(validation_rows)
    best_two = (
        validation_frame.sort_values(["rmse", "mae"], ascending=True)["model"]
        .head(2)
        .tolist()
    )

    coverage_suffix = f"p{int(round((1.0 - config.alpha) * 100))}"
    interval_details = {}
    for name in best_two:
        radius = conformal_radius(
            target_validation, predictions.loc[split.validation, name], config.alpha
        )
        lower = predictions.loc[split.test, name] - radius
        upper = predictions.loc[split.test, name] + radius
        predictions[f"{name}_lower_{coverage_suffix}"] = lower
        predictions[f"{name}_upper_{coverage_suffix}"] = upper
        interval_details[name] = {
            "conformal_radius_kwh": radius,
            "nominal_coverage": float(1.0 - config.alpha),
            "lower_column": f"{name}_lower_{coverage_suffix}",
            "upper_column": f"{name}_upper_{coverage_suffix}",
        }

    mape_eligible = bool((target_test.to_numpy(dtype=float) > 0).all())
    result_rows = []
    for name in MODEL_NAMES:
        metrics = regression_metrics(target_test, predictions.loc[split.test, name])
        row = {
            "model": name,
            "mae_kwh": metrics["mae"],
            "rmse_kwh": metrics["rmse"],
            "mape_pct": metrics["mape"] if mape_eligible else np.nan,
            "interval_model": name in best_two,
            "nominal_coverage": np.nan,
            "empirical_coverage": np.nan,
            "avg_interval_width_kwh": np.nan,
            "conformal_radius_kwh": np.nan,
        }
        if name in interval_details:
            detail = interval_details[name]
            lower = predictions.loc[split.test, detail["lower_column"]].to_numpy(dtype=float)
            upper = predictions.loc[split.test, detail["upper_column"]].to_numpy(dtype=float)
            actual = target_test.to_numpy(dtype=float)
            row.update(
                {
                    "nominal_coverage": detail["nominal_coverage"],
                    "empirical_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                    "avg_interval_width_kwh": float(np.mean(upper - lower)),
                    "conformal_radius_kwh": detail["conformal_radius_kwh"],
                }
            )
        result_rows.append(row)

    metrics_frame = pd.DataFrame(result_rows)
    metadata = {
        "dataset_label": "real",
        "cleaned_file": str(cleaned_path),
        "target": config.target_column,
        "split": _split_metadata(features.index, split),
        "split_fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
        },
        "feature_columns": list(features.columns),
        "feature_policy": "calendar fields and lagged/rolling target energy only",
        "forecast_origin": "23:00 on the day before the target day",
        "model_selection": "lowest validation RMSE, then lowest validation MAE",
        "validation_metrics": validation_frame.to_dict(orient="records"),
        "best_two_by_validation_rmse": best_two,
        "prediction_interval_method": "split conformal on absolute validation residuals",
        "alpha": config.alpha,
        "mape_eligible": mape_eligible,
        "min_test_actual_kwh": float(target_test.min()),
        "random_state": config.random_state,
        "model_params": {
            LINEAR_REGRESSION: "StandardScaler + LinearRegression defaults",
            RANDOM_FOREST: "n_estimators=300, max_depth=12, min_samples_leaf=5",
            XGBOOST: "n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9",
        },
    }
    return ForecastResult(
        metrics=metrics_frame,
        predictions=predictions.loc[split.test].copy(),
        metadata=metadata,
    )
