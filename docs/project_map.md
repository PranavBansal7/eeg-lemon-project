# Project Map

## What This Repo Is

This project is a reproducible EEG regression benchmark.

In plain English, it asks:

"If I take resting-state EEG recordings, turn them into simple numeric features, and compare a few baseline models carefully, how much useful predictive signal is there?"

The repo is strongest as a machine learning benchmark project, not as a product.

## The Big Picture

The flow is:

1. load EEG files plus metadata and target tables
2. build subject-level EEG features
3. compare a small set of feature variants and baseline models
4. save results so the run can be repeated and explained later

There is also a backend and frontend in the repo, but they are optional demo layers on top of saved model artifacts.

## What Data Comes In

The main inputs are:

- resting-state EEG files in `data/eeg/EEG_Preprocessed_BIDS_ID`
- metadata and cognitive target tables in `data/phenotype/Behavioural_Data_MPILMBB_LEMON/...`

The EEG files are paired by subject:

- `*_EO.set`
- `*_EC.set`

The target tables provide four regression targets used by the current pipeline:

- `working_memory`
- `attention`
- `executive_function`
- `intelligence`

The pipeline also adds two simple metadata fields:

- `age`
- `gender`

A subject is only used if the repo can find:

1. an EO EEG file
2. an EC EEG file
3. metadata
4. all four target values

## What EO and EC Mean

EO means "eyes open."

EC means "eyes closed."

Both are resting-state recordings, so the subject is not doing an active task. The idea is to capture baseline brain activity under two slightly different conditions.

This matters because eyes-open and eyes-closed EEG often look different, especially in the alpha range. The project keeps them separate so the model can learn from that contrast instead of averaging it away too early.

## Where the Main Logic Lives

These files matter most:

- `src/train_lemon_multitarget.py`
  Builds the base dataset from EEG, metadata, and targets.
- `src/feature_variants.py`
  Creates alternate feature sets from the base EO/EC table.
- `src/split_manifest.py`
  Creates and validates fixed train/test fold assignments by subject ID.
- `src/benchmark_v1.py`
  Runs the benchmark and writes result files.
- `run_resume_benchmark.py`
  Thin wrapper that launches the smallest, interview-friendly benchmark run.

Secondary files:

- `src/predict_my_report.py`
  Demo script for exploratory report-style input.
- `backend/app/`
  FastAPI demo layer.
- `frontend/app/`
  Next.js demo layer.

## What Preprocessing Happens

The preprocessing in this repo is intentionally simple.

For each EEG file, the code:

1. reads the EEGLAB `.set` file with MNE
2. keeps EEG channels only
3. excludes bad channels if marked
4. applies average reference

That is enough for the current spectral feature pipeline without turning the project into a large signal-processing system.

## What "Welch PSD Bandpower Features" Means

This phrase sounds technical, but the idea is simple.

The model cannot learn directly from a long raw EEG time series very easily, so the code summarizes each channel into a smaller set of numbers.

It does that in two steps:

1. Estimate how much signal energy exists at different frequencies.
2. Add up that energy inside named frequency ranges.

Welch PSD:

- "PSD" means power spectral density.
- In plain language, it estimates how strong different rhythms are in the EEG.
- "Welch" is just the averaging method used to make that estimate more stable.

Bandpower:

- Once the spectrum is estimated, the code sums power inside frequency bands.
- That creates features like delta, theta, alpha, and beta power for each channel.

Plain-English version:

"For every EEG channel, the code measures how much activity shows up in several common frequency ranges."

The current bands are:

- delta: 1-4 Hz
- theta: 4-8 Hz
- low alpha: 8-10 Hz
- alpha: 8-12 Hz
- high alpha: 10-12 Hz
- low beta: 12-20 Hz
- beta: 12-30 Hz
- high beta: 20-30 Hz

Each feature name also keeps track of whether it came from EO or EC.

## How the Base Dataset Is Built

For each usable subject, the training script:

1. computes EO bandpower features
2. computes EC bandpower features
3. appends `age` and `gender`
4. stores the four target values

That produces one row per subject.

So the final table is not a raw EEG table. It is a subject-level machine learning table.

## What the Feature Variants Mean

The repo supports multiple feature variants, but the main public set is intentionally small.

### `eo_ec_concat`

This is the clean baseline.

It keeps EO features and EC features as separate columns in the same row.

Meaning:

"Use both resting-state conditions, but do not mix them together yet."

### `eo_ec_concat_plus_regions`

This starts from the baseline above and adds regional averages such as frontal or occipital summaries.

Meaning:

"Keep the channel-level detail, but also give the model a simpler region-level view."

### `eo_ec_concat_plus_diff_plus_regions`

This adds EO-minus-EC difference features on top of the baseline and regional summaries.

Meaning:

"Let the model see the raw EO values, the raw EC values, and the explicit change between them."

This is the easiest "best current variant" to explain in interviews because it is still interpretable.

### Advanced variants

The repo also keeps some more exploratory variants, such as log-ratio, ratio-based, and asymmetry-based features.

These still work, but they are not part of the main story. They are better described as optional research extensions.

## What the Benchmark Script Does

`src/benchmark_v1.py` is the main benchmark runner.

Its job is to make model comparisons reproducible.

It does the following:

1. loads the subject-level dataset
2. checks that the expected target columns are still the ones being used
3. loads or creates fixed subject-level folds
4. builds the requested feature variant
5. trains the requested model on each fold
6. scales targets using the train fold only
7. converts predictions back to raw target space
8. computes metrics in raw target space
9. writes run metadata and CSV outputs

The default benchmark story is intentionally small:

- models: `dummy`, `ridge`, `random_forest`
- feature variants:
  `eo_ec_concat`,
  `eo_ec_concat_plus_regions`,
  `eo_ec_concat_plus_diff_plus_regions`

`run_resume_benchmark.py` is the easiest way to launch that default suite.

## Why the Split Manifest Exists

The repo saves subject-level fold assignments in `processed/splits/benchmark_v1_splits.csv`.

This helps in two ways:

1. the same person never appears in both train and test within a fold
2. future runs can reuse the same folds, which makes comparisons fairer

Without fixed splits, it would be harder to tell whether a result changed because of a real improvement or just a different random partition.

## What Models Are Compared

The simplified default suite is:

- `dummy`
  A floor. It tells you what happens if you predict something trivial like the mean.
- `ridge`
  A simple linear baseline.
- `random_forest`
  A stronger nonlinear tabular baseline.

Optional exploratory models still exist in `benchmark_v1.py`, including `elasticnet` and `hist_gb`.

## What Gets Saved

The repo saves two kinds of outputs.

### Training artifacts

These come from `src/train_lemon_multitarget.py`:

- `models/rf_model.pkl`
- `processed/features.csv`
- `processed/feature_columns.json`
- `processed/target_scaler.json`

These are mainly for the trained-model and demo paths.

### Benchmark artifacts

These come from `benchmark_v1.py`:

- `results/benchmark_v1/<run_id>/metadata.json`
- `results/benchmark_v1/<run_id>/split_summary.csv`
- `results/benchmark_v1/<run_id>/fold_results.csv`
- `results/benchmark_v1/<run_id>/summary_results.csv`

These are the most important files for explaining benchmark runs.

## What the Saved Results Currently Say

The strongest saved setup is:

- model: `random_forest`
- feature variant: `eo_ec_concat_plus_diff_plus_regions`

Current saved cross-validation results are modest:

- `attention`: mean R2 about `0.07`
- `executive_function`: mean R2 about `0.13`
- `working_memory`: near `0`
- `intelligence`: negative R2

So the honest conclusion is:

"There is some signal for attention and executive function, but the current feature set is weak for working memory and intelligence."

## How to Study the Repo in Order

If you want to understand the code top-to-bottom, read it in this order:

1. `README.md`
2. `src/train_lemon_multitarget.py`
3. `feature_variants.py`
4. `split_manifest.py`
5. `benchmark_v1.py`
6. `run_resume_benchmark.py`
7. `src/predict_my_report.py`
8. `backend/app/`
9. `frontend/app/`

That order keeps the core ML story first and the optional demo pieces second.
