# NeuroScope

## A reproducible EEG feature-engineering and regression benchmark on resting-state EO/EC data.

## Project Overview

NeuroScope is organized around a simple ML question: how much predictive signal can we recover from resting-state EEG when we keep eyes-open (EO) and eyes-closed (EC) features separate, add lightweight metadata, and benchmark the resulting feature sets carefully?

The main value of this repository is the reproducible benchmark loop:

1. build subject-level EEG features from EO/EC recordings
2. compare feature variants and baseline models on fixed subject-level splits
3. save artifacts and result summaries so the experiment is easy to rerun and explain

The FastAPI backend and Next.js frontend are still in the repo, but they are optional demo layers around saved model artifacts rather than the core project.

Current repo layout, with generated folders like `.next/` and `node_modules/` omitted:

```text
eeg-lemon-project/
├── src/
│   ├── benchmark_v1.py
│   ├── feature_variants.py
│   ├── split_manifest.py
│   ├── train_lemon_multitarget.py
│   └── predict_my_report.py
├── data/
│   ├── eeg/
│   └── phenotype/
├── processed/
│   ├── features.csv
│   ├── feature_columns.json
│   ├── target_scaler.json
│   └── splits/
├── results/
│   └── benchmark_v1/
├── models/
│   └── rf_model.pkl
├── backend/
│   └── app/
├── docs/
├── frontend/
│   └── app/
├── research_notes/
├── run_resume_benchmark.py
└── README.md
```

## Dataset and Targets

This repo expects LEMON-style resting-state EEG and phenotype data:

- EEG inputs live under `data/eeg/EEG_Preprocessed_BIDS_ID` as paired `*_EO.set` and `*_EC.set` files.
- Metadata and cognitive tables live under `data/phenotype/Behavioural_Data_MPILMBB_LEMON/...`.
- The current pipeline appends two metadata fields: `age` and `gender`.
- The current regression targets are `working_memory`, `attention`, `executive_function`, and `intelligence`.
- In code, those targets map to `TAP_WM_1`, `TAP_A_1`, `TAP_I_1`, and `WST_1`.
- A subject is only included if EO and EC files, metadata, and all four targets are present.
- Saved benchmark runs in `results/benchmark_v1/` use 200 subjects with fixed 5-fold splits of 160 train and 40 test per fold.

These targets should be treated as exploratory regression targets, not clinical outcomes.

## Core ML Pipeline

The core pipeline lives in `src/train_lemon_multitarget.py` and `src/benchmark_v1.py`.

1. Load metadata and target tables.
2. Match subjects who have both EO and EC recordings.
3. Read each EEGLAB `.set` file with MNE.
4. Keep EEG channels only and apply average reference.
5. Compute Welch PSD and integrate channel-wise bandpower from 1 to 30 Hz.
6. Build subject-level EO and EC feature tables.
7. Append metadata features `age` and `gender`.
8. Create derived feature variants for ablation and benchmarking.
9. Train regression baselines and evaluate them per target on fixed subject-level folds.
10. Save feature tables, model artifacts, split manifests, and benchmark summaries.

The current spectral feature family includes:

- `delta` (1-4 Hz)
- `theta` (4-8 Hz)
- `low_alpha` (8-10 Hz)
- `alpha` (8-12 Hz)
- `high_alpha` (10-12 Hz)
- `low_beta` (12-20 Hz)
- `beta` (12-30 Hz)
- `high_beta` (20-30 Hz)

## Feature Variants Actually Worth Discussing in Interviews

There are many implemented variants, but these are the ones most worth talking through:

- `eo_ec_concat`: the clean baseline. EO and EC are kept as separate feature subspaces instead of being averaged away.
- `eo_ec_concat_plus_diff`: adds EO-minus-EC difference features to capture state contrast explicitly.
- `eo_ec_concat_plus_regions`: adds region-level summaries across frontal, central, parietal, occipital, and temporal channels.
- `eo_ec_concat_plus_diff_plus_regions`: the best overall benchmark story in the saved runs because it combines raw EO/EC features, explicit EO-EC differences, and interpretable regional means.
- `eo_ec_concat_plus_diff_plus_regions_plus_ratios`: useful as an ablation, but it did not improve the strongest targets in saved runs.
- `eo_ec_concat_plus_diff_plus_regions_plus_ratios_plus_asymmetry`: another useful negative result; adding more handcrafted EEG heuristics did not automatically help.

Interview-wise, the strongest framing is not "I built many variants." It is "I tested increasingly structured feature families and kept the ones that were interpretable and actually useful."

## Benchmarking and Reproducibility

The benchmarking workflow is designed to be easy to rerun and audit:

- `src/split_manifest.py` creates and validates fixed subject-level fold assignments.
- `processed/splits/benchmark_v1_splits.csv` stores the reusable split manifest.
- `src/benchmark_v1.py` locks target definitions so silent target-column drift is visible.
- Target scaling is done on the train fold only, then predictions are inverse-transformed before metrics are computed in raw target space.
- Timestamped benchmark outputs are written to `results/benchmark_v1/<run_id>/`.
- Each benchmark run writes `metadata.json`, `split_summary.csv`, `fold_results.csv`, and `summary_results.csv`.
- Training artifacts for the demo/inference path are written to `models/rf_model.pkl`, `processed/features.csv`, `processed/feature_columns.json`, and `processed/target_scaler.json`.

Typical commands:

```bash
pip install -r requirements.txt
python src/train_lemon_multitarget.py
python run_resume_benchmark.py
```

`run_resume_benchmark.py` is the recommended one-command entrypoint for interviews and demos. Use `python -m src.benchmark_v1 ...` when you want to request optional exploratory models or advanced feature variants directly.

## Results Summary

The honest story is that current performance is modest, target-dependent, and still useful for comparing feature variants and model choices.

In `results/benchmark_v1/benchmark_v1_20260402T004104Z`, the favored saved configuration is `random_forest + eo_ec_concat_plus_diff_plus_regions` on 200 subjects with fixed 5-fold CV:

| Target | Mean CV R2 | Interpretation |
| --- | ---: | --- |
| `attention` | `0.072` | modest positive signal |
| `executive_function` | `0.133` | strongest target so far, but still modest |
| `working_memory` | `-0.020` | near the noise floor |
| `intelligence` | `-0.103` | poorly predicted with current features |

Plain-English takeaways from the saved runs:

- In `results/benchmark_v1/benchmark_v1_20260401T232221Z`, `random_forest` is clearly stronger than `ridge` on the current feature space. Ridge stays strongly negative across targets and public variants, while random forest reaches small positive R2 for `attention` and `executive_function`.
- `attention` and `executive_function` show modest but encouraging predictive signal for a student benchmark project.
- `working_memory` and `intelligence` remain weak or not reliably predicted with the current feature family.
- In `results/benchmark_v1/benchmark_v1_20260402T002127Z`, adding ratio features and frontal alpha asymmetry did not become the strongest main story. The simpler `eo_ec_concat_plus_diff_plus_regions` setup stayed easier to explain and at least as strong on the best targets.
- The main contribution here is disciplined feature engineering and reproducible benchmarking, not high-accuracy cognitive prediction.

### What These Results Suggest

- There is modest but non-zero signal for `attention` and `executive_function`, which is an encouraging direction for a student benchmark project.
- The current benchmark is useful for comparing feature variants and model choices, especially the decision to keep EO and EC separate and to test simple regional summaries.
- The results are not strong enough for broad cognitive prediction claims, especially for `working_memory` and `intelligence`.
- Even where predictive performance is limited, the project still demonstrates careful ML experimentation, transparent evaluation, and a benchmark setup that is easy to rerun and explain.

For a slightly deeper interpretation of the saved runs, see `docs/results_interpretation.md`.

## Optional API/UI Demo

The backend and frontend are still useful if you want a lightweight deployment demo. They show engineering breadth and initiative around the saved model artifacts, but they are interview-secondary compared with the benchmark itself.

- `backend/app/` contains a lightweight FastAPI demo layer for manual, CSV, and built-in demo predictions from saved artifacts.
- `frontend/app/` contains a simple Next.js demo UI for exercising those API routes.
- `src/predict_my_report.py` is an exploratory extension for sparse or report-like anchor inputs.
- The PDF upload route is included as a future extension. It currently returns a structured placeholder response rather than a model prediction.
- These components are best described as optional demo workflows, not the central evaluated path.

If you want to run the optional demo layers:

```bash
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
cd frontend
npm install
npm run dev -- --hostname 0.0.0.0 --port 3000
```

## Limitations

- Current predictive performance is modest and uneven across targets.
- The feature family is still mostly bandpower plus basic metadata.
- There is no external held-out test set in the current benchmark; evaluation is fixed 5-fold subject-level CV.
- The benchmark dataset is the filtered intersection of subjects with EO, EC, metadata, and all targets, not the full raw source dataset.
- Report-style prediction is an exploratory extension, not a validated sparse-sensor solution.
- The PDF workflow is a constructive placeholder for future report parsing, not a completed prediction pipeline.
- Nothing in this repo should be interpreted as clinical prediction or medical advice.

## How to Explain This Project in Interviews

You can explain the project in three sentences:

1. "I built a reproducible EEG regression benchmark on resting-state eyes-open and eyes-closed recordings."
2. "The core work was engineering subject-level spectral features, comparing EO/EC feature variants on fixed subject-level cross-validation splits, and saving manifests and run metadata so every comparison was rerunnable."
3. "The result was modest but honest: attention and executive function show some signal, working memory and intelligence are weak with the current feature set, and the repo is strongest as a disciplined benchmarking project rather than a flashy product."

If someone asks about deployment, the clean answer is that the FastAPI and Next.js pieces are lightweight demo layers built on top of the saved artifacts, not the main validated contribution.
