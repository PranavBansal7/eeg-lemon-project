# Project Map

## What This Repo Is

This repository is a reproducible EEG feature-engineering and regression benchmark.

In plain English, it asks:

"If I turn resting-state EEG into simple, explainable tabular features, how much predictive signal can I recover, and which feature design works best?"

The repo is strongest as a benchmark project:

- build subject-level EEG features from paired eyes-open and eyes-closed recordings
- compare a small set of feature variants and baseline models
- save enough metadata and outputs that the comparison can be rerun and explained later

## Big Picture

```text
EO .set file + EC .set file + metadata + targets
                    |
                    v
        match usable subjects by ID
                    |
                    v
      preprocess EEG (EEG-only, drop bads,
           average reference if possible)
                    |
                    v
      Welch PSD from 1-30 Hz for each channel
                    |
                    v
     integrate power inside named bands
      -> delta, theta, alpha, beta, ...
                    |
                    v
   create features like f3_alpha_eo, f3_alpha_ec
                    |
                    v
     build feature variant for benchmarking
                    |
                    v
  fixed subject-level CV with dummy / ridge / RF
                    |
                    v
 save fold_results.csv, summary_results.csv,
 split manifest, and metadata.json
```

There is also a backend and frontend in the repo, but they are optional demo layers on top of saved artifacts rather than the main evaluated contribution.

## What Data Comes In

The main inputs are:

- EEG files in `data/eeg/EEG_Preprocessed_BIDS_ID`
- metadata and cognitive tables in `data/phenotype/Behavioural_Data_MPILMBB_LEMON/...`

The EEG files are paired by subject:

- `*_EO.set`
- `*_EC.set`

The current regression targets are:

- `working_memory`
- `attention`
- `executive_function`
- `intelligence`

The pipeline also adds:

- `age`
- `gender`

A subject is only used if the repo can find:

1. an EO file
2. an EC file
3. metadata
4. all four target values

## What EO and EC Mean

EO means eyes open.

EC means eyes closed.

Both are resting-state recordings, but they are still meaningfully different conditions. In EEG, alpha activity often changes between EO and EC. This repo keeps EO and EC separate so the model can learn from that contrast instead of averaging it away too early.

## Where the Main Logic Lives

These files matter most:

- `src/train_lemon_multitarget.py`
  Builds the base subject-level dataset from EEG, metadata, and targets.
- `src/feature_variants.py`
  Creates alternate feature sets from the base EO/EC table.
- `src/split_manifest.py`
  Creates and validates fixed subject-level train/test folds.
- `src/benchmark_v1.py`
  Runs the benchmark and writes the main result files.
- `run_resume_benchmark.py`
  Thin wrapper for the simplest interview-friendly run.

Secondary but useful files:

- `src/predict_my_report.py`
  Exploratory sparse-input demo script.
- `backend/app/`
  FastAPI demo layer.
- `frontend/app/`
  Next.js demo layer.

## What Preprocessing Happens

The preprocessing in this repo is intentionally simple.

For each EEG file, the training code:

1. reads the EEGLAB `.set` file with MNE
2. keeps EEG channels only
3. excludes bad channels if they were marked
4. applies average reference
5. estimates the spectrum from 1 to 30 Hz with Welch PSD
6. integrates power inside named frequency bands

This is enough to support a clean, explainable spectral benchmark without turning the project into a very large signal-processing system.

## What "Welch PSD Bandpower Features" Means

This sounds technical, but the idea is simple.

The model does not work directly on a long raw EEG time series. Instead, the code summarizes each channel into a smaller set of numbers.

It does that in two steps:

1. estimate how much signal power exists at each frequency
2. add up that power inside named ranges such as alpha or beta

Plain-English version:

"For each EEG channel, the code measures how much activity appears in several common frequency ranges."

Current bands:

- delta: 1-4 Hz
- theta: 4-8 Hz
- low alpha: 8-10 Hz
- alpha: 8-12 Hz
- high alpha: 10-12 Hz
- low beta: 12-20 Hz
- beta: 12-30 Hz
- high beta: 20-30 Hz

## One Worked Feature Example

Take the feature name `f3_alpha_eo`.

Here is what it means:

1. `f3`
   Use the F3 EEG channel.
2. `alpha`
   Look only at the 8-12 Hz alpha band.
3. `eo`
   Take that value from the eyes-open recording.

So the full story is:

`sub-032_EO.set` -> preprocess EEG -> estimate Welch PSD for F3 -> integrate power from 8 to 12 Hz -> save the result as `f3_alpha_eo`

That one number becomes one column in the subject-level ML table.

## How Feature Naming Works

The base naming pattern is:

`channel_band_state`

Examples:

- `f3_alpha_eo`
- `o1_theta_ec`
- `cz_beta_eo`

Derived features follow the same logic but add a suffix that explains what changed.

Examples:

- `f3_alpha_diff`
  EO minus EC for that base feature
- `region_frontal_alpha_eo`
  frontal-region mean alpha feature for EO

This naming scheme is useful in interviews because the feature columns are readable without needing a decoder.

## What the Main Feature Variants Mean

The repo supports many variants, but the public default set is intentionally small.

### `eo_ec_concat`

Use EO features and EC features side by side.

Meaning:

"Give the model both resting-state conditions, but keep them separate."

### `eo_ec_concat_plus_regions`

Start from the baseline above and add regional averages.

Meaning:

"Keep channel detail, but also give the model simpler region-level summaries."

### `eo_ec_concat_plus_diff_plus_regions`

Add EO-minus-EC difference features on top of the baseline and regional summaries.

Meaning:

"Let the model see the raw EO values, the raw EC values, and the explicit contrast between them."

This is the easiest best-current variant to explain because it stays interpretable.

### Exploratory variants

The repo also keeps narrower or more handcrafted variants such as:

- `eo_only`
- `ec_only`
- `eo_ec_diff`
- `eo_ec_logratio`
- ratio-based features
- asymmetry-based features

These still work, but they are better described as exploratory comparisons rather than the main benchmark story.

## What the Benchmark Script Does

`src/benchmark_v1.py` is the main benchmark runner.

Its job is to make feature and model comparisons reproducible.

It does the following:

1. builds the subject-level dataset
2. checks that the expected target columns are the ones actually being used
3. loads or creates a fixed split manifest
4. builds the requested feature variant
5. trains each requested model on each fold
6. scales targets using the train fold only
7. converts predictions back to raw target space
8. computes metrics in raw target space
9. writes run metadata and CSV summaries

The default benchmark story is intentionally small:

- models: `dummy`, `ridge`, `random_forest`
- feature variants:
  `eo_ec_concat`,
  `eo_ec_concat_plus_regions`,
  `eo_ec_concat_plus_diff_plus_regions`

`run_resume_benchmark.py` is the easiest way to launch that default suite.

## Why the Split Manifest Exists

The repo saves subject-level fold assignments in `processed/splits/benchmark_v1_splits.csv`.

This matters because:

1. the same person never appears in both train and test within one fold
2. future runs can reuse the exact same folds
3. model and feature changes can be compared fairly

Without fixed splits, a result change could come from a different random partition instead of a real modeling difference.

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

These come from `src/benchmark_v1.py`:

- `results/benchmark_v1/<run_id>/metadata.json`
- `results/benchmark_v1/<run_id>/split_summary.csv`
- `results/benchmark_v1/<run_id>/fold_results.csv`
- `results/benchmark_v1/<run_id>/summary_results.csv`

These are the most important files for understanding the benchmark itself.

## What the Saved Results Currently Say

The strongest saved setup is:

- model: `random_forest`
- feature variant: `eo_ec_concat_plus_diff_plus_regions`

Current saved mean CV R2 values are:

- `attention`: about `0.07`
- `executive_function`: about `0.13`
- `working_memory`: about `-0.02`
- `intelligence`: about `-0.10`

The clean takeaway is:

"There is modest but encouraging signal for attention and executive function, while working memory and intelligence are still more limited in the current setup."

That is why the repo is best discussed as a benchmark-first project and reproducible comparison framework, not as a general cognitive prediction system.

## Good Reading Order

If you want to understand the repo top to bottom, read it in this order:

1. `README.md`
2. `src/train_lemon_multitarget.py`
3. `src/feature_variants.py`
4. `src/split_manifest.py`
5. `src/benchmark_v1.py`
6. `run_resume_benchmark.py`
7. `docs/preprocessing_cheatsheet.md`
8. `docs/feature_variant_cheatsheet.md`
9. `docs/metrics_and_model_reading_guide.md`
10. `src/predict_my_report.py`
11. `backend/app/`
12. `frontend/app/`

See also:

- `docs/README.md`
- `docs/preprocessing_cheatsheet.md`
- `docs/feature_variant_cheatsheet.md`
- `docs/ml_concepts_for_this_project.md`
- `docs/metrics_and_model_reading_guide.md`
- `docs/file_by_file_code_map.md`
