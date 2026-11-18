# Loom walkthrough script

**Target length:** about 2 minutes 30 seconds  
**Camera:** on; share the screen while keeping your camera visible.  
**Suggested screen order:** report page 1 → `src/train.py` → `validation_predictions.csv` → scorer chart.

## 0:00–0:20 | Introduction

Hi, I’m Bhuvan Teja. This is my freight-rate prediction assessment. I’ll cover what I found in the data, how I validated the model, and where the final predictions and December chart are generated.

## 0:20–0:50 | Data findings

The development data has 48,000 labeled loads from January through October 2025. The November and December validation file has 12,000 loads. The rate target is right-skewed, with a median near $2,031 and a mean near $2,374. I found 300 missing training weights and 374 missing market-index values. I impute weight from the training fold and leave market index out because it is not supplied for the December chart. There are no duplicate training rows or load IDs.

## 0:50–1:25 | Validation and model

Since these are dated loads, I used three expanding monthly backtests: January through July to predict August, then through August to predict September, then through September to predict October. Across 14,282 out-of-fold loads, pooled MAPE is 4.56 percent. October MAPE is 5.19 percent, improved from 6.37 percent in my initial model on that same month.

Nearly 99.7 to 99.8 percent of backtest lanes had earlier examples. I use only strictly prior mean and median rates per mile and counts for each lane, lane and equipment, pickup and equipment, and equipment. The model predicts log rate per mile with histogram gradient boosting and an absolute-error objective, then multiplies by distance to return a rate. I also tried market, quote, and coordinate fields; they did not generalize as well across the time folds, and the December input lacks them.

## 1:25–2:05 | Code walkthrough

In `src/train.py`, `prepare_base` builds calendar features and route groups. `add_prior_training_rates` creates historical features from earlier labeled loads only, while `add_known_history` maps those training aggregates to future rows. The script checks that the validation IDs match the supplied template, writes exactly `load_id,predicted_rate`, and fills all 31 December rates.

## 2:05–2:30 | Outputs

The provided scorer validates the 12,000 predictions and the 31 fixed December rows, then creates the candidate chart shown here and included in my report. The README has the setup and run commands. Thanks for reviewing my assessment.

## Recording checklist

- Start a Loom recording with your camera visible and screen sharing enabled.
- Keep the report and repository open in the suggested order above.
- Read naturally; the time markers are pacing guides, not lines to announce.
- After recording, confirm the share settings match the assessment’s recipient/access requirements, then add the Loom link to the submission message.
