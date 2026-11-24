# Assessment walkthrough transcript

**Voice:** External Microsoft Edge neural TTS, English (India), male Prabhat voice. Captions are timed from the speech service word-boundary data. The opening identifies this as AI narration. This is a slide video, not a camera-on Loom recording.

## Scene 1

This is an AI-narrated walkthrough of my freight-rate assessment. I’ll cover the data, forward validation, model, and final files. The objective is to estimate a posted rate for each future load using information available when that prediction is made.

## Scene 2

The development data contains forty-eight thousand loads from January through October twenty twenty-five. The final file has twelve thousand loads. Three hundred weights are missing, so I impute them within each training fold. Market index, quote signal, and coordinates are left out because they are not provided in the December inputs. There are no duplicate training IDs, and nearly all holdout lanes have earlier examples.

## Scene 3

To mimic predicting forward, I train through July and predict August, then train through August and predict September, and finally train through September and predict October. Across fourteen thousand two hundred eighty-two holdout predictions, pooled MAPE is four point five six percent and mean absolute error is one hundred two dollars and twenty-three cents. October is the hardest fold, at five point one nine percent MAPE.

## Scene 4

For each load, I calculate earlier average and median rates per mile, and prior counts, for the same lane, lane and equipment, pickup and equipment, and equipment. Training rows only see strictly earlier loads, preventing their own or future prices from leaking into the features. Efficient cumulative means and grouped medians build this history.

## Scene 5

The estimator is histogram gradient boosting with an absolute-error objective. It learns log rate per mile from load details, calendar information, route, equipment, and historical comparisons. At inference, the predicted rate per mile is converted back to dollars by multiplying by the load distance.

## Scene 6

In the training script, prepare base creates date and route fields. Add prior training rates builds past-only history features. Make model defines the imputers, category encoding, and estimator. Predict rates is shared by backtesting and final inference. The feature table is computed once and reused across folds and final training.

## Scene 7

The final model is refit on all labeled loads. It writes the twelve-thousand validation predictions in template order and fills all thirty-one December chart rows. The scorer accepted both files and created the chart shown here. The repository includes the code, README, report, predictions, chart, and this video. The GitHub link is on screen. Thank you for reviewing my assessment.
