# NeuroScope

A reproducible EEG feature-engineering and regression benchmark on resting-state EO/EC data.

## Project Overview

NeuroScope is a benchmark-first ML project built around a simple question:

How much predictive signal can we recover from resting-state EEG when eyes-open (EO) and eyes-closed (EC) recordings are kept separate, converted into clear tabular features, and evaluated on fixed subject-level splits?

The project is strongest in three areas:

- a clear and explainable EEG feature-engineering pipeline
- reproducible benchmarking across a small, readable model and feature suite
- honest evaluation with saved run artifacts that are easy to inspect and compare

The repository also includes an optional demo API and UI layer, but the benchmark is the main validated path.

## Core Technical Contributions

- Built a subject-level EEG regression pipeline from paired EO/EC EEGLAB recordings, phenotype tables, and cognitive targets.
- Engineered Welch PSD bandpower features plus lightweight metadata features (`age`, `gender`) for tabular regression.
- Defined a compact public feature ladder centered on `eo_ec_concat`, `eo_ec_concat_plus_regions`, and `eo_ec_concat_plus_diff_plus_regions`.
- Preserved broader ratio and asymmetry ideas as exploratory extensions rather than the main public story.
- Added reproducibility safeguards including fixed subject-level split manifests, target-definition locking, train-fold-only target scaling, raw-space metrics, and timestamped run outputs.
- Exposed the main benchmark through a simple entrypoint, [`run_resume_benchmark.py`](run_resume_benchmark.py), so the default path is easy to run and explain.

## Dataset and Task

The repo expects LEMON-style resting-state EEG and phenotype data:

- EEG files under `data/eeg/EEG_Preprocessed_BIDS_ID` as paired `*_EO.set` and `*_EC.set`
- metadata and cognitive tables under `data/phenotype/Behavioural_Data_MPILMBB_LEMON/...`
- current regression targets: `working_memory`, `attention`, `executive_function`, and `intelligence`

Subjects are included only when EO and EC recordings, metadata, and all four targets are present.

Saved benchmark runs in `results/benchmark_v1/` use 200 subjects with fixed 5-fold subject-level splits. These targets are best interpreted as exploratory regression targets rather than clinical outcomes.

## Benchmark Pipeline

The main ML workflow lives in [`src/train_lemon_multitarget.py`](src/train_lemon_multitarget.py), [`src/feature_variants.py`](src/feature_variants.py), [`src/split_manifest.py`](src/split_manifest.py), and [`src/benchmark_v1.py`](src/benchmark_v1.py).

The public benchmark path is:

1. match subjects with EO, EC, metadata, and targets
2. preprocess each EEG recording and compute Welch PSD bandpower features from 1 to 30 Hz
3. build a subject-level EO/EC feature table plus `age` and `gender`
4. benchmark `dummy`, `ridge`, and `random_forest` across the public feature variants
5. save fold-level metrics, summary CSVs, split summaries, and run metadata

## Main Results

Saved runs show a benchmark-first result: modest but encouraging signal on selected targets, a clearer fit for `random_forest` than `ridge` in the current feature space, and useful feature/model comparisons that stay easy to explain.

Using `results/benchmark_v1/benchmark_v1_20260402T004104Z`, the leading saved configuration is `random_forest + eo_ec_concat_plus_diff_plus_regions`:

| Target | Mean CV R2 | Takeaway |
| --- | ---: | --- |
| `attention` | `0.072` | modest but encouraging signal |
| `executive_function` | `0.133` | strongest target so far, still modest |
| `working_memory` | `-0.020` | currently limited signal in this setup |
| `intelligence` | `-0.103` | currently limited signal in this setup |

Across the saved comparison runs:

- `random_forest` performs better than `ridge` on the current feature family
- `attention` and `executive_function` show the most encouraging predictive signal
- `working_memory` and `intelligence` remain more limited with the present features
- more complex ratio and asymmetry variants did not become a stronger main story than the simpler EO/EC plus regions plus difference setup

The strongest value lies in reproducible comparison and disciplined evaluation rather than headline predictive accuracy.

## Repo Structure

```text
eeg-lemon-project/
├── src/
│   ├── train_lemon_multitarget.py
│   ├── benchmark_v1.py
│   ├── feature_variants.py
│   ├── split_manifest.py
│   └── predict_my_report.py
├── backend/
│   └── app/
├── frontend/
│   └── app/
├── docs/
├── processed/
├── results/
│   └── benchmark_v1/
├── models/
└── run_resume_benchmark.py
```

## Quick Start

Install the Python dependencies and run the main benchmark path:

```bash
pip install -r requirements.txt
python run_resume_benchmark.py
```

That command runs the public default suite and writes outputs to `results/benchmark_v1/<run_id>/`.

For the fuller benchmark CLI:

```bash
python -m src.benchmark_v1 --help
```

## Optional Demo Layer

The repo also includes a lightweight FastAPI + Next.js demo layer around saved benchmark artifacts.

- [`backend/app/`](backend/app/) exposes manual, CSV, and built-in demo prediction routes
- [`frontend/app/`](frontend/app/) provides a simple UI for those routes
- [`src/predict_my_report.py`](src/predict_my_report.py) is an exploratory sparse-input extension
- the PDF route is included as a future extension and currently stops at structured acknowledgement

This demo layer adds useful engineering breadth, but it is secondary to the benchmark itself.

## Scope and Limitations

- Current predictive performance is modest and target-dependent.
- The feature family is still mostly bandpower plus basic metadata.
- The benchmark uses fixed 5-fold subject-level cross-validation rather than an external held-out test set.
- The working dataset is the filtered intersection of subjects with EO, EC, metadata, and all four targets.
- The demo API/UI and sparse-input paths are useful extensions, but they are not the central evaluated contribution.
- Nothing in this repo should be interpreted as clinical prediction or medical advice.

## Learn More

For deeper understanding and interview prep, start with:

- [`docs/README.md`](docs/README.md) for the main study guide
- [`docs/project_map.md`](docs/project_map.md) for the repo-wide beginner walkthrough
- [`docs/mental_model_of_data_flow.md`](docs/mental_model_of_data_flow.md) for the end-to-end pipeline story
- [`docs/interview_notes.md`](docs/interview_notes.md) for ML/SWE interview framing
- [`docs/web_layer_for_interviews.md`](docs/web_layer_for_interviews.md) for the optional demo layer
