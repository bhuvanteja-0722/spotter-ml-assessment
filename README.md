# Freight Rate Prediction Challenge

This repository trains a freight-rate model, creates the required 12,000-row validation submission, and fills the fixed December forecast inputs. The reproducible run and its time-based holdout result are documented below.

## Setup and run

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/train.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The training command writes `validation_predictions.csv` with exactly `load_id,predicted_rate`, fills `data/december_chart_inputs.csv`, and writes holdout metrics to `outputs/validation_metrics.json`. The scorer validates the final predictions and fixed December inputs, then creates `scorer_results/candidate_december.png`.

## Validation approach

The labeled development data contains 48,000 loads from January 1 through October 31, 2025. The latest full month, October (4,853 loads), is held out as a forward-looking validation set; January–September rows are used for that fit. This preserves time order and avoids random-split leakage. The October holdout scored **$125.21 MAE**, **$647.88 RMSE**, and **6.37% MAPE**. The final model is then refit using all 48,000 labeled rows before predicting the November–December validation loads.

## Data preparation and model

The target is `posted_rate`. The model uses distance, weight, date-derived calendar features, pickup, delivery, and equipment. These are the features shared by the labeled data, validation rows, and the fixed December chart inputs. `market_index`, `quote_signal`, and coordinates are deliberately excluded because the December chart file does not provide them. Missing weights are median-imputed using the training fold; unseen route names are supported by the one-hot encoder. A histogram gradient-boosting regressor is trained on `log1p(posted_rate)` and predictions are transformed back to dollars and clipped above zero.

Data checks found 300 missing training weights (0.63%) and 374 missing training market-index values (0.78%). The latter field is not used. The training set has no duplicate rows or duplicate load IDs. The validation submission is checked against the supplied template's row count and ID order before it is saved.

## Submission files

- `validation_predictions.csv` - final 12,000-row submission.
- `data/december_chart_inputs.csv` - the 31 fixed December rows with predicted rates.
- `scorer_results/candidate_december.png` - the fixed December chart created by the supplied scorer.
- `outputs/assessment_report.pdf` - validation and methodology report with the scorer chart.
- `outputs/assessment_report.docx` - editable report source.
- `outputs/loom_walkthrough_script.md` - a timed 2–3 minute recording script.

The scorer reports structural validity only; Spotter calculates the final hidden validation metric after submission.
