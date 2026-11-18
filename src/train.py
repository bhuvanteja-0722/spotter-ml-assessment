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
]
CATEGORICAL = ["pickup", "delivery", "equipment"]
FEATURES = BASE_NUMERIC + HISTORY_NUMERIC + CATEGORICAL
HISTORY_KEYS = {
    "lane": "lane_key",
    "lane_equipment": "lane_equipment_key",
    "origin_equipment": "origin_equipment_key",
    "equipment": "equipment",
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


def add_prior_training_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add expanding historical rates using only strictly earlier labeled rows."""
    result = prepare_base(frame).sort_values(["date", "load_id"]).copy()
    rate_per_mile = result[TARGET] / result["distance"].replace(0, np.nan)
    for name, key in HISTORY_KEYS.items():
        grouped_rpm = rate_per_mile.groupby(result[key], sort=False)
        result[f"hist_{name}_rpm"] = grouped_rpm.transform(
            lambda values: values.shift().expanding(min_periods=1).mean()
        )
        result[f"hist_{name}_count"] = result.groupby(key, sort=False).cumcount().astype(float)
    return result[FEATURES]


def add_known_history(history: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Map historical rate aggregates from labeled history onto future rows."""
    prior = prepare_base(history)
    result = prepare_base(frame)
    rpm = prior[TARGET] / prior["distance"].replace(0, np.nan)
    for name, key in HISTORY_KEYS.items():
        stats = pd.DataFrame({"group": prior[key], "rpm": rpm}).groupby("group")["rpm"].agg(["mean", "count"])
        result[f"hist_{name}_rpm"] = result[key].map(stats["mean"])
        result[f"hist_{name}_count"] = result[key].map(stats["count"]).fillna(0).astype(float)
    return result[FEATURES]


def make_model() -> Pipeline:
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
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.07,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("features", preprocessing), ("model", estimator)])


def summarize(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100),
    }


def rolling_validation(train: pd.DataFrame) -> dict[str, object]:
    """Use three expanding monthly folds to evaluate future-month generalization."""
    dates = pd.to_datetime(train["date"])
    months = sorted(dates.dt.to_period("M").unique())
    validation_months = months[-3:]
    folds: list[dict[str, object]] = []
    all_actual: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []

    for month in validation_months:
        fit_mask = dates.dt.to_period("M") < month
        valid_mask = dates.dt.to_period("M") == month
        fit = train.loc[fit_mask].sort_values(["date", "load_id"]).reset_index(drop=True)
        valid = train.loc[valid_mask].reset_index(drop=True)
        model = make_model()
        model.fit(add_prior_training_rates(fit), np.log1p(fit[TARGET] / fit["distance"]))
        predicted = np.expm1(model.predict(add_known_history(fit, valid))) * valid["distance"].to_numpy()
        actual = valid[TARGET].to_numpy()
        predicted = np.maximum(predicted, 0.01)
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

    train = pd.read_csv(args.data_dir / "train_test.csv").sort_values(["date", "load_id"]).reset_index(drop=True)
    validation = pd.read_csv(args.data_dir / "validation.csv")
    template = pd.read_csv(args.data_dir / "validation_predictions_template.csv")
    chart_inputs = pd.read_csv(args.data_dir / "december_chart_inputs.csv")

    metrics = rolling_validation(train)
    final_model = make_model()
    final_model.fit(add_prior_training_rates(train), np.log1p(train[TARGET] / train["distance"]))

    validation_rates = np.expm1(final_model.predict(add_known_history(train, validation))) * validation["distance"].to_numpy()
    predictions = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": np.maximum(validation_rates, 0.01)})
    predictions["predicted_rate"] = predictions["predicted_rate"].round(2)
    if len(template) != len(predictions) or not template["load_id"].equals(predictions["load_id"]):
        raise ValueError("Validation prediction rows do not match the supplied template")
    submission_path = ROOT / "validation_predictions.csv"
    predictions.to_csv(submission_path, index=False)

    chart_rates = np.expm1(final_model.predict(add_known_history(train, chart_inputs))) * chart_inputs["distance"].to_numpy()
    chart_inputs["predicted_rate"] = np.maximum(chart_rates, 0.01).round(2)
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
