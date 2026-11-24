# Assessment walkthrough narration

**Narration:** synthetic voice generated with macOS text-to-speech. The video is a slide walkthrough and does not include a live camera recording.

## Scene 1

This is an AI-narrated walkthrough of my freight-rate prediction assessment. I’ll explain what is in the data, how I validated the model, and how each prediction is produced. The goal is a practical rate estimate for every future load, with a reproducible process and an honest measure of historical performance.

## Scene 2

The labeled development file contains forty-eight thousand loads from January through October twenty twenty-five. The final validation file contains twelve thousand loads. I found three hundred missing weights and three hundred seventy-four missing market-index values. Weight is imputed inside each training fold. Market index, quote signal, and coordinates are excluded because they are not available in the fixed December inputs. There are no duplicate training IDs. Nearly all lanes in the holdout months had earlier labeled examples.

## Scene 3

I used expanding monthly validation to mimic predicting forward in time. The first model trains through July and predicts August. The second trains through August and predicts September. The third trains through September and predicts October. Across fourteen thousand two hundred eighty-two future-month predictions, pooled mean absolute error is one hundred two dollars and twenty-three cents, and MAPE is four point five six percent. October is the hardest month, at five point one nine percent MAPE.

## Scene 4

The most useful history comes from earlier comparable loads. For each row, I calculate the mean and median rate per mile, and the number of prior examples, for its pickup-to-delivery lane, lane and equipment, pickup and equipment, and equipment. Each training row uses strictly earlier loads only, so its own rate and future rates cannot leak into its features. The calculations use vectorized cumulative means and grouped expanding medians.

## Scene 5

The model is histogram gradient boosting with an absolute-error objective. It learns log rate per mile from distance, weight, calendar details, route, equipment, and route-history summaries. At prediction time, I convert the model output back to a rate per mile and multiply by the load distance. This separates price intensity from shipment size and produces a dollar rate.

## Scene 6

In the training script, prepare base creates calendar and route fields. Add prior training rates builds the point-in-time history features. Make model defines the imputation, category encoding, and regressor. Predict rates is shared by validation and final inference, so both outputs use the same conversion. The feature matrix is created once and reused across the validation folds and final fit.

## Scene 7

Finally, the model is refit on all forty-eight thousand labeled rows. It writes a twelve-thousand-row validation file in the required ID order and fills all thirty-one December chart rows. The supplied scorer accepted both files and generated the fixed December chart shown here. The README includes the full setup and run instructions. Thank you for reviewing my assessment.
