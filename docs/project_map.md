# Project Map

This doc explains the repository from top to bottom in plain English.

If `README.md` gives you the public story, this file gives you the beginner-friendly internal map.

## What This Repo Is

This repository is a reproducible EEG feature-engineering and regression benchmark.

In simple terms, it asks:

If I turn resting-state EEG into clear tabular features, which feature design and baseline model recover the most useful signal?

## Core vs Optional

### Core benchmark path

This is the main project story:

- paired EO and EC EEG files
- preprocessing
- Welch PSD bandpower features
- subject-level feature table
- feature variants
- fixed subject-level folds
- model comparison
- saved benchmark outputs

Main files:

- `src/train_lemon_multitarget.py`
- `src/feature_variants.py`
- `src/split_manifest.py`
- `src/benchmark_v1.py`
- `run_resume_benchmark.py`

### Optional demo path

This is secondary:

- `src/predict_my_report.py`
- `backend/app/`
- `frontend/app/`

These files show useful engineering breadth, but they are not the main evaluated contribution.

## Big Picture

```text
EO .set file + EC .set file + metadata + targets
                    |
                    v
        match usable subjects by ID
                    |
                    v
      preprocess EEG and compute spectra
                    |
                    v
      integrate power inside named bands
                    |
                    v
   create features like f3_alpha_eo, f3_alpha_ec
                    |
                    v
   build feature variants for fair comparison
                    |
                    v
   run fixed subject-level folds with baseline models
                    |
                    v
 save summary_results.csv, fold_results.csv,
 split summary, and metadata.json
```

## What Data Comes In

The main inputs are:

- EEG files in `data/eeg/EEG_Preprocessed_BIDS_ID`
- metadata and cognitive tables in `data/phenotype/Behavioural_Data_MPILMBB_LEMON/...`

Each usable subject needs:

1. one EO file
2. one EC file
3. metadata
4. all four target values

The four current targets are:

- `working_memory`
- `attention`
- `executive_function`
- `intelligence`

The pipeline also adds:

- `age`
- `gender`

## What EO and EC Mean

EO means eyes open.

EC means eyes closed.

Both are resting-state recordings, but they are not the same condition. This matters because some EEG activity changes between EO and EC, especially alpha-related activity.

That is why the repo keeps EO and EC separate instead of averaging them too early.

## One Subject Becomes One Row

A good way to picture the repo is to follow one example subject.

Imagine the code finds:

- `sub-032_EO.set`
- `sub-032_EC.set`
- one metadata row with age and gender
- one row for each target

At this point the training script is trying to turn that subject into one clean ML row.

That row will contain:

- many EO features such as `f3_alpha_eo`
- many EC features such as `f3_alpha_ec`
- metadata columns like `age` and `gender`
- target values such as `attention`

So one subject becomes one row in the final dataset.

## What Preprocessing Happens

For each EEG file, the training script does a small, explainable preprocessing sequence:

1. read the EEGLAB `.set` file with MNE
2. keep EEG channels only
3. exclude channels marked as bad
4. apply average reference
5. estimate the spectrum from 1 to 30 Hz with Welch PSD
6. integrate power inside named frequency bands

This is intentionally simple. The goal is a clean spectral benchmark, not a huge preprocessing system.

## What "Welch PSD Bandpower Features" Means

This phrase sounds more technical than it really is.

You can think of it like this:

1. Welch PSD estimates how much signal power exists at each frequency
2. bandpower adds up the power inside a named range such as alpha or beta

So the model does not see the full raw time series.

It sees a table of summary numbers like:

- how much alpha power F3 had in EO
- how much beta power O1 had in EC

## One Concrete Feature Example

Take the feature `f3_alpha_eo`.

It means:

- `f3`
  use the F3 channel
- `alpha`
  use the 8-12 Hz alpha band
- `eo`
  use the eyes-open recording

So the flow is:

`sub-032_EO.set` -> preprocess EEG -> estimate Welch PSD for F3 -> integrate 8-12 Hz power -> save that number as `f3_alpha_eo`

That one value becomes one column in the subject-level table.

## What the Feature Variants Mean

The repo has many supported variants, but the public default set is intentionally small.

### `eo_ec_concat`

Give the model EO and EC features side by side.

Plain-English meaning:

"Use both resting-state conditions, but keep them separate."

### `eo_ec_concat_plus_regions`

Start from the baseline above and add region-level averages.

Plain-English meaning:

"Keep the channel detail, but also give the model simpler regional summaries."

### `eo_ec_concat_plus_diff_plus_regions`

Add EO-minus-EC differences on top of the baseline and region summaries.

Plain-English meaning:

"Let the model see EO, EC, and the explicit contrast between them."

This is the cleanest best-current feature story for interviews.

## What the Benchmark Script Does

`src/benchmark_v1.py` is the main benchmark runner.

At a high level, it:

1. builds the subject-level dataset
2. checks that the expected target columns are the ones actually being used
3. loads or creates a fixed split manifest
4. builds the requested feature variant
5. trains the requested model on each fold
6. scales targets using the train fold only
7. converts predictions back to raw target space
8. computes metrics
9. writes metadata and CSV outputs

The default public suite is deliberately small:

- models: `dummy`, `ridge`, `random_forest`
- feature variants:
  `eo_ec_concat`,
  `eo_ec_concat_plus_regions`,
  `eo_ec_concat_plus_diff_plus_regions`

## Why the Split Manifest Exists

The split manifest lives at `processed/splits/benchmark_v1_splits.csv`.

It matters because:

1. the same subject never appears in both train and test within one fold
2. later runs can reuse the exact same folds
3. result changes are easier to trust because the partition did not drift

Concrete picture:

- in one saved run, each fold has 160 train subjects and 40 test subjects
- those assignments are written to disk
- later model and feature comparisons reuse those exact assignments

Without this step, a result change might just come from a new random split.

## What Gets Saved

### Training artifacts

These come from `src/train_lemon_multitarget.py`:

- `models/rf_model.pkl`
- `processed/features.csv`
- `processed/feature_columns.json`
- `processed/target_scaler.json`

These are mainly for the saved-model and demo paths.

### Benchmark artifacts

These come from `src/benchmark_v1.py`:

- `results/benchmark_v1/<run_id>/metadata.json`
- `results/benchmark_v1/<run_id>/split_summary.csv`
- `results/benchmark_v1/<run_id>/fold_results.csv`
- `results/benchmark_v1/<run_id>/summary_results.csv`

These are the most important files for understanding the benchmark itself.

## What the Current Results Say

The cleanest saved story is:

- `random_forest` is stronger than `ridge` on the current feature space
- `attention` and `executive_function` show modest but encouraging signal
- `working_memory` and `intelligence` are still more limited with the current feature family

So the repo is best explained as:

- a reproducible benchmark-first ML project
- strong on feature engineering, evaluation design, and comparison discipline
- best interpreted as a benchmark-first comparison rather than a clinical or broad prediction system

## What To Read Next

If this file made sense, the next best docs are:

1. `docs/mental_model_of_data_flow.md`
2. `docs/deep_walkthrough_train_lemon_multitarget.md`
3. `docs/deep_walkthrough_benchmark_v1.md`
4. `docs/interview_notes.md`
