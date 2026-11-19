"""Train a time-aware freight-rate model and generate assessment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
TARGET = "posted_rate"
BASE_NUMERIC = [
    "distance", "weight", "year", "month", "day_of_week", "day_of_year",
    "month_sin", "month_cos",
]
HISTORY_NUMERIC = [
    "hist_lane_rpm", "hist_lane_count", "hist_lane_equipment_rpm",
    "hist_lane_equipment_count", "hist_origin_equipment_rpm", "hist_origin_equipment_count",
    "hist_equipment_rpm", "hist_equipment_count",
    "hist_lane_rpm_median", "hist_lane_equipment_rpm_median",
    "hist_origin_equipment_rpm_median", "hist_equipment_rpm_median",
]
CATEGORICAL = ["pickup", "delivery", "equipment"]
FEATURES = BASE_NUMERIC + HISTORY_NUMERIC + CATEGORICAL
HISTORY_KEYS = {
    "lane": "lane_key",
    "lane_equipment": "lane_equipment_key",
    "origin_equipment": "origin_equipment_key",
    "equipment": "equipment",
}
MODEL_PARAMETERS = {
    "loss": "absolute_error",
    "learning_rate": 0.07,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def prepare_base(frame: pd.DataFrame) -> pd.DataFrame:
    """Create fields shared by labeled rows, validation, and chart inputs."""
    result = frame.copy()
    dates = pd.to_datetime(result["date"], errors="coerce")
    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_year"] = dates.dt.dayofyear
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    result["distance"] = pd.to_numeric(result["distance"], errors="coerce")
    result["lane_key"] = result["pickup"].astype(str) + "|" + result["delivery"].astype(str)
    result["lane_equipment_key"] = result["lane_key"] + "|" + result["equipment"].astype(str)
    result["origin_equipment_key"] = result["pickup"].astype(str) + "|" + result["equipment"].astype(str)
    return result


def add_prior_training_rates(
    frame: pd.DataFrame, *, already_sorted: bool = False
) -> pd.DataFrame:
    """Add expanding historical rates using only strictly earlier labeled rows."""
    result = prepare_base(frame)
    if not already_sorted:
        result = result.sort_values(["date", "load_id"])
    result = result.copy()
    rate_per_mile = result[TARGET] / result["distance"].replace(0, np.nan)
    for name, key in HISTORY_KEYS.items():
        grouped_rpm = rate_per_mile.groupby(result[key], sort=False)
        prior_count = result.groupby(key, sort=False).cumcount().astype(float)
        prior_sum = grouped_rpm.cumsum() - rate_per_mile
        result[f"hist_{name}_rpm"] = prior_sum.div(prior_count.where(prior_count > 0))

        # Use pandas' grouped expanding implementation instead of a Python
        # callback per lane; this is materially faster on the assessment data.
        shifted_rpm = grouped_rpm.shift()
        expanding_median = shifted_rpm.groupby(result[key], sort=False).expanding(min_periods=1).median()
        result[f"hist_{name}_rpm_median"] = (
            expanding_median.reset_index(level=0, drop=True).reindex(result.index)
        )
        result[f"hist_{name}_count"] = prior_count
    return result[FEATURES].reset_index(drop=True)


def add_known_history(history: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Give future rows aggregates learned from labeled history."""
    prior = prepare_base(history)
    result = prepare_base(frame)
    rpm = prior[TARGET] / prior["distance"].replace(0, np.nan)
    for name, key in HISTORY_KEYS.items():
        stats = (
            pd.DataFrame({"group": prior[key], "rpm": rpm})
            .groupby("group", sort=False)["rpm"]
            .agg(["mean", "median", "count"])
        )
        result[f"hist_{name}_rpm"] = result[key].map(stats["mean"])
        result[f"hist_{name}_rpm_median"] = result[key].map(stats["median"])
        result[f"hist_{name}_count"] = result[key].map(stats["count"]).fillna(0).astype(float)
    return result[FEATURES]


def predict_rates(
    model: Pipeline, history: pd.DataFrame, loads: pd.DataFrame
) -> np.ndarray:
    """Predict dollar rates by converting predicted rate-per-mile back to dollars."""
    features = add_known_history(history, loads)
    predicted_rate_per_mile = np.exp(model.predict(features))
    return np.maximum(predicted_rate_per_mile * loads["distance"].to_numpy(), 0.01)


def make_model() -> Pipeline:
    """Build preprocessing plus a robust, nonlinear tabular regressor."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), BASE_NUMERIC + HISTORY_NUMERIC),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
    return Pipeline([("features", preprocessing), ("model", estimator)])


def summarize(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100),
    }


def rolling_validation(
    train: pd.DataFrame, training_features: pd.DataFrame | None = None
) -> dict[str, object]:
    """Score future months; passed features must align with chronologically sorted rows."""
    if training_features is None:
        train = train.sort_values(["date", "load_id"]).reset_index(drop=True)
        training_features = add_prior_training_rates(train, already_sorted=True)
    if len(training_features) != len(train):
        raise ValueError("Precomputed training features must align with the training rows")

    dates = pd.to_datetime(train["date"])
    months = sorted(dates.dt.to_period("M").unique())
    validation_months = months[-3:]
    folds: list[dict[str, object]] = []
    all_actual: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []

    for month in validation_months:
        fit_mask = (dates.dt.to_period("M") < month).to_numpy()
        valid_mask = dates.dt.to_period("M") == month
        fit = train.loc[fit_mask]
        valid = train.loc[valid_mask]
        model = make_model()
        model.fit(training_features.loc[fit_mask], np.log(fit[TARGET] / fit["distance"]))
        predicted = predict_rates(model, fit, valid)
        actual = valid[TARGET].to_numpy()
        folds.append({"month": str(month), **summarize(actual, predicted)})
        all_actual.append(actual)
        all_predicted.append(predicted)

    pooled_actual = np.concatenate(all_actual)
    pooled_predicted = np.concatenate(all_predicted)
    return {
        "strategy": "expanding-window monthly validation; each fold predicts the next full month",
        "folds": folds,
        "pooled": summarize(pooled_actual, pooled_predicted),
        "mean_fold_mape_percent": float(np.mean([fold["mape_percent"] for fold in folds])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train_test.csv")
    train = train.sort_values(["date", "load_id"]).reset_index(drop=True)
    validation = pd.read_csv(args.data_dir / "validation.csv")
    template = pd.read_csv(args.data_dir / "validation_predictions_template.csv")
    chart_inputs = pd.read_csv(args.data_dir / "december_chart_inputs.csv")

    # Every labeled row's history uses only strictly earlier loads, so this
    # feature matrix is safe to reuse in all time-ordered folds and final fit.
    training_features = add_prior_training_rates(train, already_sorted=True)
    metrics = rolling_validation(train, training_features)
    final_model = make_model()
    final_model.fit(training_features, np.log(train[TARGET] / train["distance"]))

    validation_rates = predict_rates(final_model, train, validation)
    predictions = pd.DataFrame(
        {"load_id": validation["load_id"], "predicted_rate": validation_rates.round(2)}
    )
    if len(template) != len(predictions) or not template["load_id"].equals(predictions["load_id"]):
        raise ValueError("Validation prediction rows do not match the supplied template")
    submission_path = ROOT / "validation_predictions.csv"
    predictions.to_csv(submission_path, index=False)

    chart_inputs["predicted_rate"] = predict_rates(
        final_model, train, chart_inputs
    ).round(2)
    chart_path = args.data_dir / "december_chart_inputs.csv"
    chart_inputs.to_csv(chart_path, index=False)

    with (args.output_dir / "validation_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")

    pooled = metrics["pooled"]
    print(
        f"3-fold pooled OOS: MAE=${pooled['mae']:.2f}, "
        f"RMSE=${pooled['rmse']:.2f}, MAPE={pooled['mape_percent']:.2f}%"
    )
    print(f"Wrote {len(predictions):,} validation rows to {submission_path}")
    print(f"Wrote {len(chart_inputs):,} December chart rows to {chart_path}")


if __name__ == "__main__":
    main()
