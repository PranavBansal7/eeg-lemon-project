# Mental Model Of Data Flow

This doc explains the whole repository like a story.

The goal is to help you picture the pipeline clearly enough that you can explain it without getting lost in file names.

## The Short Story

The repo takes:

- raw EEG files
- metadata
- target tables

and turns them into:

- a subject-level feature table
- reproducible train/test folds
- model predictions
- saved benchmark metrics and metadata

## The Whole Flow In One Line

```text
raw EEG -> preprocessing -> Welch PSD -> bandpower -> feature table -> feature variants -> fixed folds -> model training -> held-out predictions -> metrics -> saved benchmark outputs
```

## One Running Example

Imagine one subject:

- `sub-032_EO.set`
- `sub-032_EC.set`

And imagine one target:

- `attention`

We will follow that subject through the pipeline.

## Step 1: Start With Raw Files And Tables

The repo begins with two kinds of data:

- EEG files
- tabular subject information

For `sub-032`, the code wants:

- one EO EEG file
- one EC EEG file
- one metadata row
- one target row for each of the four targets

If any of those parts are missing, the subject is not used.

## Step 2: Match A Usable Subject

At this point the code is trying to answer:

"Do I have everything needed to build one full training example for this subject?"

If the answer is yes, `sub-032` stays in the dataset.

If the answer is no, `sub-032` is skipped.

## Step 3: Preprocess The EO Recording

Now the training script opens:

- `sub-032_EO.set`

It then:

1. keeps EEG channels only
2. excludes bad channels
3. applies average reference

You can think of this as:

"Get the EEG into a clean, consistent form before extracting features."

## Step 4: Turn The EO Recording Into Spectral Features

Now the script estimates Welch PSD from 1 to 30 Hz.

This gives a frequency-based view of the signal.

Then it adds up power inside named bands such as:

- delta
- theta
- alpha
- beta

## Step 5: Create One Concrete Feature

Take the feature:

- `f3_alpha_eo`

This means:

- `f3`
  the F3 channel
- `alpha`
  the 8-12 Hz alpha band
- `eo`
  the eyes-open recording

So the code has turned a long EEG time series into one table value:

"How much alpha-band power did F3 have in EO?"

## Step 6: Do The Same For EC

The script repeats the same process for:

- `sub-032_EC.set`

That produces features such as:

- `f3_alpha_ec`
- `o1_beta_ec`
- `cz_theta_ec`

Now the subject has two feature groups:

- EO features
- EC features

## Step 7: Build One Subject Row

At this point the script is trying to make one clean tabular row.

For `sub-032`, that row contains:

- EO bandpower features
- EC bandpower features
- `age`
- `gender`

It also keeps the target values, including:

- `attention`

So one subject becomes:

- one input row
- one target row

## Step 8: Build A Feature Variant

The base table is not the end of the story.

The benchmark then creates feature variants from that base row.

Examples:

- `eo_ec_concat`
  keep EO and EC side by side
- `eo_ec_concat_plus_regions`
  add regional summaries
- `eo_ec_concat_plus_diff_plus_regions`
  also add EO-minus-EC differences

You can think of this as:

"Take the same subject row and look at it through slightly different feature-design lenses."

## Step 9: Put The Subject Into A Fold

Now the benchmark looks at the saved split manifest.

Imagine `sub-032` lands in:

- fold 3
- test split

That means:

- `sub-032` will not be used to train the model in fold 3
- the model in fold 3 must predict `sub-032` using patterns learned from other subjects

This is how the benchmark tests held-out generalization.

## Step 10: Train On Other Subjects

For that fold, the model trains on the train subjects only.

The benchmark may try:

- `dummy`
- `ridge`
- `random_forest`

Each model sees:

- the chosen feature variant
- the train subjects
- scaled training targets

The test subject stays unseen until prediction time.

## Step 11: Predict One Target For The Held-Out Subject

Now imagine the model predicts:

- `attention` for `sub-032`

That prediction is compared against the true held-out attention value for `sub-032`.

The benchmark repeats this for:

- every test subject in the fold
- every fold
- every selected model
- every selected feature variant

## Step 12: Turn Predictions Into Metrics

Once all held-out predictions are collected, the benchmark computes:

- `R2`
- `RMSE`
- `MAE`
- `Pearson correlation`

These are saved:

- per fold in `fold_results.csv`
- averaged across folds in `summary_results.csv`

## Step 13: Save The Run Outputs

Each benchmark run gets its own folder under:

- `results/benchmark_v1/<run_id>/`

That folder stores:

- `metadata.json`
- `split_summary.csv`
- `fold_results.csv`
- `summary_results.csv`

This is what makes the project feel reproducible instead of one-off.

## What One Fold Really Means

It helps to picture one fold very concretely.

For one saved run:

- 160 subjects are in train
- 40 subjects are in test

The model learns from the 160 train subjects and is scored on the 40 unseen test subjects.

Then the benchmark rotates to the next fold and repeats.

## What The Final Result Means

When you read a result like:

- `attention`
- `random_forest`
- `eo_ec_concat_plus_diff_plus_regions`
- `R2 = 0.072`

the mental translation is:

"Using that feature design and model, the benchmark recovered a modest but non-zero amount of attention-related signal on held-out subjects."

## Where The Demo Layer Fits

The backend and frontend come later.

They do not retrain the benchmark.

They reuse saved artifacts such as:

- `models/rf_model.pkl`
- `processed/feature_columns.json`

So the demo layer is best seen as:

"A lightweight wrapper around saved benchmark artifacts."

## Final Mental Picture

If you only remember one story, remember this one:

One subject starts as two raw EEG files plus tabular metadata.

The code turns that subject into readable bandpower features, places the subject into fixed held-out folds, compares multiple model and feature choices fairly, and saves the results in a way that is easy to inspect later.
