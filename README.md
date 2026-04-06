# NeuroScope

A reproducible EEG feature-engineering and regression benchmark on resting-state EO/EC data.

## Project Overview

NeuroScope is an ML benchmark project built around a simple question: how much predictive signal can we recover from resting-state EEG when eyes-open (EO) and eyes-closed (EC) recordings are kept separate, converted into interpretable tabular features, and evaluated on fixed subject-level splits?

The repo is centered on three things:

- subject-level EEG feature engineering from paired EO/EC recordings
- reproducible benchmarking across a small, explainable model and feature suite
- saved artifacts and summaries that make runs easy to rerun, compare, and discuss

The backend and frontend remain in the repository as optional demo layers around saved artifacts, but the benchmark is the main validated path.

## Core Technical Contributions

- Built a subject-level EEG regression pipeline from paired EO/EC EEGLAB recordings, phenotype tables, and cognitive targets.
- Engineered Welch PSD bandpower features plus lightweight metadata (`age`, `gender`) for tabular regression.
- Defined a compact public feature suite that is easy to explain in interviews: `eo_ec_concat`, `eo_ec_concat_plus_regions`, and `eo_ec_concat_plus_diff_plus_regions`.
- Preserved more complex ratio and asymmetry variants as exploratory extensions rather than the main benchmark story.
- Added reproducibility safeguards including fixed subject-level split manifests, target-definition locking, train-fold-only target scaling, raw-space metric reporting, and timestamped run outputs.
- Exposed the benchmark through one clear entrypoint, [`run_resume_benchmark.py`](run_resume_benchmark.py), so a first-time reader can run the main path with a single command.

## Dataset and Task

The repo expects LEMON-style resting-state EEG and phenotype data:

- EEG files live under `data/eeg/EEG_Preprocessed_BIDS_ID` as paired `*_EO.set` and `*_EC.set` files.
- Metadata and cognitive tables live under `data/phenotype/Behavioural_Data_MPILMBB_LEMON/...`.
- The current regression targets are `working_memory`, `attention`, `executive_function`, and `intelligence`.
- In code, those targets map to `TAP_WM_1`, `TAP_A_1`, `TAP_I_1`, and `WST_1`.
- Subjects are included only when EO and EC recordings, metadata, and all four targets are present.

Saved benchmark runs in `results/benchmark_v1/` use 200 subjects with fixed 5-fold subject-level splits of 160 train and 40 test per fold. These targets should be interpreted as exploratory regression targets rather than clinical outcomes.

## Benchmark Pipeline

The main ML workflow lives in [`src/train_lemon_multitarget.py`](src/train_lemon_multitarget.py), [`src/feature_variants.py`](src/feature_variants.py), [`src/split_manifest.py`](src/split_manifest.py), and [`src/benchmark_v1.py`](src/benchmark_v1.py).

1. Load metadata and target tables, then match subjects who have both EO and EC recordings.
2. Read each EEGLAB `.set` file with MNE, keep EEG channels, apply average reference, and compute Welch PSD bandpower features from 1 to 30 Hz.
3. Build subject-level EO and EC feature tables, then append `age` and `gender`.
4. Compare the interview-friendly default suite of models: `dummy`, `ridge`, and `random_forest`.
5. Benchmark the public feature variants on fixed subject-level folds and compute metrics in raw target space.
6. Save reproducibility artifacts to `results/benchmark_v1/<run_id>/`, including `metadata.json`, `split_summary.csv`, `fold_results.csv`, and `summary_results.csv`.

For exploratory extensions, `python -m src.benchmark_v1 --help` exposes optional models and advanced feature variants without changing the default benchmark story.

## Main Results

Saved runs show a benchmark-first result: modest but encouraging signal on a subset of targets, clear separation between more and less competitive baseline choices, and useful comparisons between simple and more elaborate feature variants.

Using `results/benchmark_v1/benchmark_v1_20260402T004104Z`, the leading saved configuration is `random_forest + eo_ec_concat_plus_diff_plus_regions`:

| Target | Mean CV R2 | Takeaway |
| --- | ---: | --- |
| `attention` | `0.072` | modest but encouraging signal |
| `executive_function` | `0.133` | strongest target so far, still modest |
| `working_memory` | `-0.020` | limited signal in the current setup |
| `intelligence` | `-0.103` | currently limited signal with these features |

Across the saved comparison runs:

- `random_forest` is a better fit than `ridge` for the current feature space, based on `results/benchmark_v1/benchmark_v1_20260401T232221Z`.
- `attention` and `executive_function` show the most encouraging predictive signal in the current benchmark.
- `working_memory` and `intelligence` remain more limited with the present feature family.
- More complex ratio and asymmetry variants did not become a stronger main story than the simpler EO/EC plus regions plus difference setup, based on `results/benchmark_v1/benchmark_v1_20260402T002127Z`.

The strongest value of the project comes from reproducible feature and model comparison rather than headline predictive accuracy. For deeper result interpretation, see [`docs/results_interpretation.md`](docs/results_interpretation.md).

## Repo Structure

```text
eeg-lemon-project/
├── src/
│   ├── train_lemon_multitarget.py
│   ├── benchmark_v1.py
│   ├── feature_variants.py
│   ├── split_manifest.py
│   └── predict_my_report.py
├── data/
├── processed/
├── results/
│   └── benchmark_v1/
├── models/
├── backend/
│   └── app/
├── frontend/
│   └── app/
├── docs/
└── run_resume_benchmark.py
```

## Quick Start

Recommended benchmark path:

1. Place the expected EEG and phenotype files under `data/eeg/EEG_Preprocessed_BIDS_ID` and `data/phenotype/Behavioural_Data_MPILMBB_LEMON/...`.
2. Install the Python dependencies.
3. Run the benchmark through the simplified entrypoint.

```bash
pip install -r requirements.txt
python run_resume_benchmark.py
```

That command runs the interview-friendly default suite and writes results to `results/benchmark_v1/<run_id>/`.

If you want the more configurable benchmark CLI, use:

```bash
python -m src.benchmark_v1 --help
```

## Optional API/UI Demo Layer

The repository also includes a lightweight demo stack that sits on top of saved model artifacts. It is useful for showing engineering breadth, but it is secondary to the benchmark itself.

- [`backend/app/`](backend/app/) contains a FastAPI demo API for manual, CSV, and built-in demo prediction workflows.
- [`frontend/app/`](frontend/app/) contains a small Next.js UI that exercises those API routes.
- [`src/predict_my_report.py`](src/predict_my_report.py) is an exploratory sparse-input extension.
- The PDF upload route is included as a future extension and currently stops at structured file acknowledgement rather than model prediction.

For local frontend/backend routing details and env vars such as `BACKEND_INTERNAL_URL` and `NEXT_PUBLIC_API_BASE_URL`, see [`docs/web_layer_for_interviews.md`](docs/web_layer_for_interviews.md).

If you want to run the demo stack, first generate the saved model artifacts:

```bash
python src/train_lemon_multitarget.py
```

Then start the backend:

```bash
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

And the frontend:

```bash
cd frontend
npm install
npm run dev -- --hostname 0.0.0.0 --port 3000
```

## Scope and Limitations

- Current predictive performance is modest and target-dependent.
- The feature family is still mostly bandpower plus basic metadata.
- The benchmark uses fixed 5-fold subject-level cross-validation rather than an external held-out test set.
- The working dataset is the filtered intersection of subjects with EO, EC, metadata, and all four targets.
- The web layer and sparse-input paths are useful demo or exploratory extensions, but they are not the central evaluated contribution.
- Nothing in this repo should be interpreted as clinical prediction or medical advice.

Deeper repo walkthroughs, result interpretation, and interview-prep material live in [`docs/README.md`](docs/README.md), [`docs/project_map.md`](docs/project_map.md), [`docs/results_interpretation.md`](docs/results_interpretation.md), [`docs/interview_notes.md`](docs/interview_notes.md), and [`docs/web_layer_for_interviews.md`](docs/web_layer_for_interviews.md).
