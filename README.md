# Freight Rate Prediction Challenge

This repository builds a model to estimate a load's posted freight rate. It trains on the labeled development file, checks future-month performance using time-ordered validation, writes predictions for every validation load, and fills the supplied December chart inputs.

## Results

The current model was evaluated on 14,282 out-of-fold loads from three future-month holdouts:

| Holdout month | Loads | MAE | MAPE |
| --- | ---: | ---: | ---: |
| August 2025 | 4,759 | $91.19 | 4.21% |
| September 2025 | 4,670 | $104.64 | 4.27% |
| October 2025 | 4,853 | $110.75 | 5.19% |
| **All holdouts** | **14,282** | **$102.23** | **4.56%** |

MAE is the average absolute dollar difference between a prediction and the posted rate. MAPE is the average absolute error as a percentage of the posted rate; lower is better. Results are estimates from historical holdouts, not a guarantee of performance on the hidden assessment labels.

## How a prediction is made

For each load, the model receives its distance, weight, date, pickup, delivery, and equipment. It also receives historical average and median rates per mile, plus the number of earlier observations, for four groups:

- The same pickup-to-delivery lane
- The same lane and equipment
- The same pickup and equipment
- The same equipment

The model learns to predict the logarithm of the rate per mile. At prediction time, it converts that estimate back to a rate per mile and multiplies by the load distance to produce a dollar rate. In plain terms, it combines the load's known details with what earlier comparable loads cost.

The estimator is a histogram gradient-boosting regressor with an absolute-error loss. It learns nonlinear relationships among the inputs and is less driven by unusually large errors than a squared-error objective. The model is a tree ensemble, so it does not have one coefficient per feature; the feature-generation steps above describe the main signals it uses.

### Data preparation

- Calendar fields are derived from the load date, including day of week and seasonal sine/cosine values.
- Missing numeric inputs are filled using the median learned from the applicable training fold.
- Categorical inputs are filled with the most common training value and one-hot encoded. Categories not seen during training are accepted safely.
- Historical rate features for a training row use only loads dated earlier than that row. The validation and December rows use aggregates from the labeled development data.
- Historical means are calculated with cumulative sums, and historical medians use grouped expanding calculations. The resulting past-only feature table is built once and reused across validation folds and the final fit.

The target is `posted_rate`. Training uses `log(posted_rate / distance)` so the model learns a per-mile price before converting back to a total rate. `market_index`, `quote_signal`, and coordinates are excluded because they are not present in the December prediction inputs.

## Validation design

The development data contains 48,000 labeled loads dated January 1 through October 31, 2025. A random split would allow earlier and later loads to mix, which would not reflect the task of predicting future loads. Instead, validation uses three expanding-window folds:

1. Train on January–July; predict August.
2. Train on January–August; predict September.
3. Train on January–September; predict October.

Each row's historical rate features also use only earlier loads. The final model is then refit on all 48,000 labeled rows before predicting the assessment validation and December chart inputs.

## Data checks

The development file has 300 missing weights (0.63%) and 374 missing `market_index` values (0.78%). Weight is retained and imputed within each training fold; `market_index` is not used. The development data has no duplicate rows or duplicate `load_id` values. Before saving the final submission, the training script checks its row count and `load_id` order against the supplied template.

## Run the project

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Training writes:

- `validation_predictions.csv` with exactly `load_id,predicted_rate` and one row per validation load.
- `data/december_chart_inputs.csv` with the predicted rate filled for each December chart row.
- `outputs/validation_metrics.json` with the fold and pooled holdout metrics.

The scorer checks the submission and December input structure and creates `scorer_results/candidate_december.png`. It does not calculate the hidden validation score; Spotter calculates that after submission.

## Project files

- `src/train.py` — data preparation, historical features, model, time-based validation, and prediction generation.
- `score.py` — supplied assessment scorer.
- `data/train_test.csv` — labeled development data.
- `data/validation.csv` — loads requiring final predictions.
- `data/validation_predictions_template.csv` — required load IDs and output format.
- `data/december_chart_inputs.csv` — fixed December chart rows; training fills its `predicted_rate` column.
- `outputs/assessment_report.pdf` and `outputs/assessment_report.docx` — assessment report.
- `outputs/loom_walkthrough_script.md` — outline for the requested short video; record and attach the Loom video separately.

The scorer requirements are listed in `requirements.txt`. The final hidden assessment result depends on Spotter's data and scoring, so the backtest and structural checks should be read as evidence about this solution, not a promise of a particular award or ranking.
