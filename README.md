# Spotter freight-rate assessment

This repository contains a reproducible regression workflow for estimating `posted_rate` from the supplied freight-load data. It generates the validation submission, December chart input predictions, a forecast chart, and a time-ordered holdout report.

## Results

The workflow reserves the latest month in `train-test.csv` as a temporal holdout before fitting the final model on all labeled rows. On the October 2025 holdout (4,853 loads), it reached **$125.21 MAE**, **$647.88 RMSE**, and **6.37% MAPE**. The metrics are recorded in [`outputs/validation-metrics.json`](outputs/validation-metrics.json). The requested validation predictions are in [`outputs/validation-predictions.csv`](outputs/validation-predictions.csv), and December predictions are in [`outputs/december-chart-predictions.csv`](outputs/december-chart-predictions.csv).

The forecast chart is available at [`outputs/december-rate-forecast.png`](outputs/december-rate-forecast.png).

## Run

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python src/train.py
```

The command reads the four provided CSVs from `data/` and writes outputs to `outputs/`. Pass `--data-dir` and `--output-dir` to use different locations.

## Modeling approach

- A histogram gradient-boosting regressor models the log of the positive rate target, then converts predictions back to dollars per load.
- Numeric inputs include distance, weight, and calendar features. Pickup, delivery, and equipment use one-hot encoding; unseen categories are supported.
- Missing weights are imputed from the training median. The model only uses fields available in both the labeled training data and December chart input, so it does not depend on November-only market or quote fields that are absent from the chart file.
- Validation metrics use the final calendar month in the labeled data as a forward-looking holdout. The provided validation labels are not present in the supplied files, so this is an estimate of generalization and not the hidden assessment score.

## Data

The `data/` directory contains the assessment's four supplied CSV inputs. The `outputs/` directory contains generated, row-aligned predictions and the chart. The supplied validation template's `load_id` order is checked before writing the submission.
