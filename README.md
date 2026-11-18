# Freight Rate Prediction Challenge

This repository trains a freight-rate model, creates the required 12,000-row validation submission, and fills the fixed December forecast inputs. The reproducible run and its expanding-window backtest are documented below.

## Setup and run

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/train.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The training command writes `validation_predictions.csv` with exactly `load_id,predicted_rate`, fills `data/december_chart_inputs.csv`, and writes holdout metrics to `outputs/validation_metrics.json`. The scorer validates the final predictions and fixed December inputs, then creates `scorer_results/candidate_december.png`.

## Validation approach

The labeled development data contains 48,000 loads from January 1 through October 31, 2025. Validation uses three expanding monthly folds: train through July and predict August, train through August and predict September, then train through September and predict October. This keeps each fold forward-looking. Across 14,282 out-of-fold predictions, the model scores **$102.23 MAE**, **$625.84 RMSE**, and **4.56% MAPE**. The October fold scores **5.19% MAPE**, down from 6.37% for the initial model on that same month. The final model is refit using all 48,000 labeled rows before predicting November–December.

## Data preparation and model

The target is `posted_rate`. The model uses distance, weight, date-derived calendar features, pickup, delivery, and equipment. These fields are available in training, validation, and December inputs. It also uses strictly prior mean and median rates per mile and observation counts for the lane, lane plus equipment, pickup plus equipment, and equipment. The route history is built in date order, so each training row only uses earlier labeled loads. Historical means use cumulative sums and medians use pandas grouped expanding calculations, avoiding a per-lane Python callback while preserving the same past-only features. Nearly 99.7–99.8% of lanes in the three backtest months had appeared earlier. A histogram gradient-boosting regressor with an absolute-error objective is trained on `log(posted_rate / distance)`; predictions are converted back to dollars per load. Missing numeric features are median-imputed within each training fold, and unseen route categories are supported by the one-hot encoder. `market_index`, `quote_signal`, and coordinates were excluded because the December chart file lacks them; a temporal feature comparison also found those fields did not generalize as well.

Data checks found 300 missing training weights (0.63%) and 374 missing training market-index values (0.78%). The latter field is not used. The training set has no duplicate rows or duplicate load IDs. The validation submission is checked against the supplied template's row count and ID order before it is saved.

## Submission files

- `validation_predictions.csv` - final 12,000-row submission.
- `data/december_chart_inputs.csv` - the 31 fixed December rows with predicted rates.
- `scorer_results/candidate_december.png` - the fixed December chart created by the supplied scorer.
- `outputs/assessment_report.pdf` - validation and methodology report with the scorer chart.
- `outputs/assessment_report.docx` - editable report source.
- `outputs/loom_walkthrough_script.md` - a timed 2–3 minute recording script.

The scorer reports structural validity only; Spotter calculates the final hidden validation metric after submission.
