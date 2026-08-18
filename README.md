# NeuroScope

NeuroScope is a reproducible EEG feature-engineering and regression benchmark for paired resting-state eyes-open (EO) and eyes-closed (EC) recordings.

## Overview

The project converts paired EEGLAB recordings and phenotype tables into a subject-level tabular dataset, then compares feature variants and regression models using fixed subject-level cross-validation splits. It is designed for transparent, repeatable experimentation rather than clinical use.

## Pipeline

1. Match subjects with EO and EC recordings, metadata, and cognitive targets.
2. Keep EEG channels, apply average reference, and estimate Welch PSD bandpower from 1–30 Hz.
3. Build EO/EC features, regional summaries, EO-minus-EC features, and basic metadata features.
4. Evaluate `dummy`, `ridge`, and `random_forest` regressors on fixed folds.
5. Save fold-level metrics, aggregate summaries, split information, and run metadata.

The benchmark targets are `working_memory`, `attention`, `executive_function`, and `intelligence`.

## Repository layout

```text
src/
  train_lemon_multitarget.py  # data preparation and training artifacts
  benchmark_v1.py             # reproducible benchmark runner
  feature_variants.py         # EEG feature variants
  split_manifest.py           # subject-level split management
backend/app/                  # optional FastAPI prediction service
frontend/app/                 # optional Next.js interface
results/benchmark_v1/         # versioned benchmark outputs
```

## Data

Place the required dataset locally; it is intentionally excluded from version control. The expected layout is:

```text
data/
  eeg/EEG_Preprocessed_BIDS_ID/    # paired *_EO.set and *_EC.set files
  phenotype/Behavioural_Data_MPILMBB_LEMON/
    META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv
    Cognitive_Test_Battery_LEMON/
```

Use the dataset only under its applicable terms and permissions.

## Run the benchmark

```bash
pip install -r requirements.txt
python -m src.benchmark_v1
```

The default suite evaluates the three baseline models across the EO/EC feature variants and writes a timestamped directory under `results/benchmark_v1/`. Run `python -m src.benchmark_v1 --help` to see the available options.

To create the artifacts used by the optional prediction service:

```bash
python -m src.train_lemon_multitarget
```

## Optional web application

Start the API from `backend/` after installing its dependencies:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then start the frontend from `frontend/`:

```bash
npm install
npm run dev
```

By default, the frontend proxies `/api` requests to `http://127.0.0.1:8000`. Set `BACKEND_INTERNAL_URL` or `NEXT_PUBLIC_API_BASE_URL` when using a different backend address.

## Limitations

- Predictive performance is exploratory and target-dependent.
- Evaluation uses fixed subject-level cross-validation rather than an external test set.
- The current features are spectral bandpower and basic metadata; they are not a clinical diagnostic system.

Nothing in this repository is medical advice or a clinical prediction tool.
