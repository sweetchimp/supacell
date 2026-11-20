import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .forecasting import (
    XGBOOST,
    ForecastConfig,
    build_day_ahead_features,
    conformal_radius,
    load_cleaned_energy,
    regression_metrics,
)

DEGRADATION_LEVELS = (
    {"name": "full", "kind": "full", "description": "Clean feature history"},
    {"name": "missing_5pct", "kind": "missing", "fraction": 0.05, "seed": 105, "description": "5% of history hours randomly unavailable"},
    {"name": "missing_15pct", "kind": "missing", "fraction": 0.15, "seed": 115, "description": "15% of history hours randomly unavailable"},
    {"name": "missing_30pct", "kind": "missing", "fraction": 0.30, "seed": 130, "description": "30% of history hours randomly unavailable"},
    {"name": "missing_50pct", "kind": "missing", "fraction": 0.50, "seed": 150, "description": "50% of history hours randomly unavailable"},
    {"name": "missing_80pct", "kind": "missing", "fraction": 0.80, "seed": 180, "description": "80% of history hours randomly unavailable"},
    {"name": "history_2y", "kind": "history", "days": 730, "description": "History truncated to 2 years before validation"},
    {"name": "history_1y", "kind": "history", "days": 365, "description": "History truncated to 1 year before validation"},
    {"name": "history_6mo", "kind": "history", "days": 183, "description": "History truncated to 6 months before validation"},
    {"name": "history_3mo", "kind": "history", "days": 90, "description": "History truncated to 3 months before validation"},
    {"name": "no_solar_data", "kind": "noop", "description": "Solar data marked unavailable; model never used solar"},
    {"name": "no_weather_features", "kind": "noop", "description": "Weather features marked unavailable; model never used weather"},
)


@dataclass(frozen=True)
class DegradationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importances: pd.DataFrame
    metadata: dict


def apply_degradation(clean_history, level, validation_start):
    history = clean_history.copy()
    kind = level["kind"]
    if kind in ("full", "noop", "missing"):
        return history
    if kind == "history":
        cutoff = pd.Timestamp(validation_start) - pd.Timedelta(days=level["days"])
        history[history.index < cutoff] = np.nan
        return history
    raise ValueError(f"unknown degradation kind: {kind}")


def is_lag_feature(column):
    return (
        column.startswith("lag_")
        or column.startswith("origin_lag_")
        or column.startswith("origin_roll_")
    )


def lag_feature_columns(features):
    return [column for column in features.columns if is_lag_feature(column)]


def apply_feature_missingness(features, fraction, seed):
    rng = np.random.default_rng(seed)
    mask = rng.random(len(features)) < fraction
    degraded = features.copy()
    degraded.loc[mask, lag_feature_columns(features)] = np.nan
    degraded.attrs["missing_mask"] = mask
    return degraded


def fill_degraded_features(features, train_index):
    columns = lag_feature_columns(features)
    if not columns:
        return features.dropna()
    filled = features.copy()
    filled[columns] = filled[columns].ffill()
    medians = features.loc[train_index, columns].median()
    filled[columns] = filled[columns].fillna(medians)
    return filled.dropna()


def split_from_bounds(index, bounds):
    normalized = pd.DatetimeIndex(index).normalize()
    train_end = pd.Timestamp(bounds["train"]["end"]).normalize()
    validation_start = pd.Timestamp(bounds["validation"]["start"]).normalize()
    validation_end = pd.Timestamp(bounds["validation"]["end"]).normalize()
    test_start = pd.Timestamp(bounds["test"]["start"]).normalize()

    train_index = index[normalized <= train_end]
    validation_index = index[
        (normalized >= validation_start) & (normalized <= validation_end)
    ]
    test_index = index[normalized >= test_start]
    if len(train_index) == 0 or len(validation_index) == 0 or len(test_index) == 0:
        raise ValueError("degraded split has an empty part")
    return train_index, validation_index, test_index


def build_best_model(metadata, config):
    best = metadata["best_two_by_validation_rmse"][0]
    if best != XGBOOST:
        raise ValueError(f"degradation experiment expects {XGBOOST}; found {best}")
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=0,
    )


def build_user_message(level, row):
    base_messages = {
        "full": "Full feature history available; operating normally.",
        "missing_5pct": "Minor gaps in history were simulated.",
        "missing_15pct": "Moderate gaps in history were simulated.",
        "missing_30pct": "Substantial gaps in history were simulated.",
        "missing_50pct": "Severe gaps in history were simulated.",
        "missing_80pct": "Extreme gaps in history were simulated.",
        "history_2y": "Only two years of history are available.",
        "history_1y": "Only one year of history is available.",
        "history_6mo": "Only six months of history are available.",
        "history_3mo": "Only three months of history are available.",
        "no_solar_data": "Solar data is not used by this model.",
        "no_weather_features": "Weather features are not used by this model.",
    }
    change = row["mae_change_pct"]
    if abs(change) < 1.0:
        trend = "MAE changed by less than 1% versus full data"
    elif change > 0:
        trend = f"MAE increased {change:.1f}% versus full data"
    else:
        trend = f"MAE decreased {abs(change):.1f}% versus full data"
    return (
        f"{base_messages[level['name']]} Measured {trend} on {row['test_rows']} scored hours; "
        f"{row['nominal_coverage']:.0%} interval coverage {row['empirical_coverage']:.1%}; "
        f"mean width {row['avg_interval_width_kwh']:.3f} kWh."
    )


def run_degradation_experiment(cleaned_path, metadata_path):
    with open(metadata_path, encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    config = ForecastConfig(
        train_fraction=source_metadata["split_fractions"]["train"],
        validation_fraction=source_metadata["split_fractions"]["validation"],
        alpha=source_metadata["alpha"],
        random_state=source_metadata["random_state"],
    )
    bounds = source_metadata["split"]
    energy = load_cleaned_energy(cleaned_path, config)
    clean_history = energy.ffill()

    metric_rows = []
    prediction_frames = []
    importance_rows = []
    for level in DEGRADATION_LEVELS:
        history = apply_degradation(
            clean_history,
            level,
            bounds["validation"]["start"],
        )
        features, target = build_day_ahead_features(
            history, target=energy, fill_missing=True
        )
        if level["kind"] == "missing":
            features = apply_feature_missingness(
                features,
                level["fraction"],
                level.get("seed", config.random_state),
            )
        train_index, validation_index, test_index = split_from_bounds(
            features.index, bounds
        )
        if level["kind"] == "missing":
            features = fill_degraded_features(features, train_index)
            target = target.loc[features.index]
            train_index, validation_index, test_index = split_from_bounds(
                features.index, bounds
            )

        features_train = features.loc[train_index]
        features_validation = features.loc[validation_index]
        features_test = features.loc[test_index]
        target_validation = target.loc[validation_index]
        target_test = target.loc[test_index]

        model = build_best_model(source_metadata, config)
        model.fit(features_train, target.loc[train_index])
        importances = pd.Series(
            model.feature_importances_, index=features.columns
        ).sort_values(ascending=False)
        lag_columns = lag_feature_columns(features)
        importance_rows.append(
            {
                "level": level["name"],
                "lag_feature_importance_share": float(importances.loc[lag_columns].sum()),
                "calendar_feature_importance_share": float(
                    importances.drop(index=lag_columns).sum()
                ),
                "top_feature": str(importances.index[0]),
                "top_feature_importance": float(importances.iloc[0]),
            }
        )
        validation_prediction = np.asarray(
            model.predict(features_validation), dtype=float
        )
        test_prediction = np.asarray(model.predict(features_test), dtype=float)
        radius = conformal_radius(target_validation, validation_prediction, config.alpha)
        lower = test_prediction - radius
        upper = test_prediction + radius
        actual = target_test.to_numpy(dtype=float)
        metrics = regression_metrics(actual, test_prediction)

        metric_rows.append(
            {
                "level": level["name"],
                "description": level["description"],
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "mae_kwh": metrics["mae"],
                "rmse_kwh": metrics["rmse"],
                "mape_pct": metrics["mape"] if bool((actual > 0).all()) else np.nan,
                "nominal_coverage": float(1.0 - config.alpha),
                "empirical_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                "avg_interval_width_kwh": float(np.mean(upper - lower)),
                "conformal_radius_kwh": radius,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "level": level["name"],
                    "model": XGBOOST,
                    "actual_kwh": actual,
                    "predicted_kwh": test_prediction,
                    "lower_kwh": lower,
                    "upper_kwh": upper,
                },
                index=test_index,
            )
        )

    metrics_frame = pd.DataFrame(metric_rows)
    full_row = metrics_frame.iloc[0]
    metrics_frame["mae_change_pct"] = (
        100.0 * (metrics_frame["mae_kwh"] - full_row["mae_kwh"]) / full_row["mae_kwh"]
    )
    metrics_frame["width_change_pct"] = (
        100.0
        * (metrics_frame["avg_interval_width_kwh"] - full_row["avg_interval_width_kwh"])
        / full_row["avg_interval_width_kwh"]
    )
    metrics_frame["coverage_delta_pp"] = (
        100.0
        * (metrics_frame["empirical_coverage"] - full_row["empirical_coverage"])
    )
    metrics_frame["user_message"] = [
        build_user_message(level, row)
        for level, row in zip(DEGRADATION_LEVELS, metrics_frame.to_dict(orient="records"))
    ]

    metadata = {
        "dataset_label": "real",
        "cleaned_file": str(cleaned_path),
        "source_forecast_metadata": str(metadata_path),
        "model": XGBOOST,
        "split": bounds,
        "alpha": config.alpha,
        "random_state": config.random_state,
        "levels": list(DEGRADATION_LEVELS),
        "feature_policy": "calendar fields and lagged/rolling target energy only",
        "feature_gap_fill": "forward fill from prior feature row; first affected row uses training-split median",
        "feature_importance_measure": "XGBoost feature_importances_",
        "solar_weather_note": "Solar and weather degradation levels are deliberate no-ops because the selected model never used those inputs.",
    }
    return DegradationResult(
        metrics=metrics_frame,
        predictions=pd.concat(prediction_frames),
        feature_importances=pd.DataFrame(importance_rows),
        metadata=metadata,
    )
