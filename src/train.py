"""Train a freight-rate model and create the requested prediction artifacts."""

from __future__ import annotations

import argparse
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
NUMERIC = ["distance", "weight", "year", "month", "day_of_week", "day_of_year", "month_sin", "month_cos"]
CATEGORICAL = ["pickup", "delivery", "equipment"]
FEATURES = NUMERIC + CATEGORICAL


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features available in both the labeled data and chart inputs."""
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
    return result[FEATURES]


def make_model() -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.07,
        max_iter=260,
        max_leaf_nodes=25,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("features", prep), ("model", estimator)])


def positive_predictions(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray(model.predict(prepare_features(frame)), dtype=float)
    return np.maximum(predictions, 0.01)


def temporal_validation(train: pd.DataFrame) -> dict[str, float]:
    """Measure generalization on the final training month, without shuffling time."""
    dates = pd.to_datetime(train["date"])
    validation_month = dates.dt.to_period("M").max()
    holdout = dates.dt.to_period("M") == validation_month
    model = make_model()
    model.fit(prepare_features(train.loc[~holdout]), np.log1p(train.loc[~holdout, TARGET]))
    predictions = np.expm1(model.predict(prepare_features(train.loc[holdout])))
    actual = train.loc[holdout, TARGET].to_numpy()
    return {
        "holdout_month": str(validation_month),
        "rows": int(holdout.sum()),
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "mape_percent": float(np.mean(np.abs((actual - predictions) / np.maximum(actual, 1))) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train_test.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    template = pd.read_csv(args.data_dir / "validation_predictions_template.csv")
    chart_inputs = pd.read_csv(args.data_dir / "december_chart_inputs.csv")

    metrics = temporal_validation(train)
    final_model = make_model()
    final_model.fit(prepare_features(train), np.log1p(train[TARGET]))

    predictions = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": np.expm1(final_model.predict(prepare_features(validation)))})
    predictions["predicted_rate"] = predictions["predicted_rate"].clip(lower=0.01).round(2)
    if len(template) != len(predictions) or not template["load_id"].equals(predictions["load_id"]):
        raise ValueError("Validation prediction rows do not match the supplied template")
    submission_path = ROOT / "validation_predictions.csv"
    predictions.to_csv(submission_path, index=False)

    chart_inputs["predicted_rate"] = np.expm1(final_model.predict(prepare_features(chart_inputs))).clip(min=0.01).round(2)
    chart_inputs.to_csv(args.data_dir / "december_chart_inputs.csv", index=False)
    with (args.output_dir / "validation_metrics.json").open("w", encoding="utf-8") as stream:
        import json

        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    print(f"Temporal holdout ({metrics['holdout_month']}): MAE=${metrics['mae']:.2f}, RMSE=${metrics['rmse']:.2f}, MAPE={metrics['mape_percent']:.2f}%")
    print(f"Wrote {len(predictions):,} validation rows to {submission_path}")
    print(f"Wrote {len(chart_inputs):,} December chart rows to {args.data_dir / 'december_chart_inputs.csv'}")


if __name__ == "__main__":
    main()
