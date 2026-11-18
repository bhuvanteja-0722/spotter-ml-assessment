# Loom walkthrough script

**Target length:** about 2 minutes 30 seconds  
**Camera:** on; use screen share for the repository and report.  
**Suggested screen order:** report page 1 → `src/train.py` → `validation_predictions.csv` → scorer chart.

## 0:00–0:20 | Introduction

Hi, I’m Bhuvan Teja. This is my freight-rate prediction assessment. I’ll show how I validated the model, the data-quality decisions I made, and the final validation and December outputs.

## 0:20–0:50 | Data findings

The labeled data contains 48,000 loads from January through October 2025, and the final validation set contains 12,000 November and December loads. Posted rates are right-skewed: the median is about $2,031, while the mean is about $2,374. I found 300 missing training weights and 374 missing market-index values. I median-impute weight within each training fit. I exclude market index and quote signal because the December chart inputs do not contain those fields. There were no duplicate training rows or load IDs.

## 0:50–1:25 | Validation and model

Because the data is time ordered, I held out all of October and trained the evaluation model on January through September. This avoids leaking future rows into training and better represents prediction on later loads. On 4,853 October loads, the model achieved $125.21 mean absolute error and 6.37 percent mean absolute percentage error. RMSE was $647.88, which is more sensitive to a small number of large rate errors.

I chose histogram gradient boosting for nonlinear relationships in distance, weight, route, equipment, and date features. I train against the log of the positive rate to reduce the effect of the long right tail. After evaluation, I refit on all 48,000 labeled rows for the final predictions.

## 1:25–2:05 | Code walkthrough

In `src/train.py`, `prepare_features` creates calendar features and keeps only inputs available in both validation and the December chart data. The pipeline imputes numeric values and one-hot encodes pickup, delivery, and equipment. The script checks that the prediction IDs match the supplied template, writes exactly `load_id,predicted_rate` to `validation_predictions.csv`, and fills all 31 December rates.

## 2:05–2:30 | Outputs

The provided scorer validates all 12,000 predictions and checks that the December file has the required route, equipment, weight, and dates. It generated the chart shown here, which is also included in my report. The repository README has the setup and run commands. Thanks for reviewing my assessment.

## Recording checklist

- Start a Loom recording with the camera visible and screen sharing enabled.
- Keep the report and repository open in the suggested order above.
- Read naturally; the time markers are pacing guides, not lines to announce.
- After recording, confirm that the share settings match the assessment’s recipient/access requirements, then add the Loom link to the submission message.
