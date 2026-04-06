# Docs Guide

This folder is the main learning and interview-prep layer for the repository.

Use it when you want to move from:

- "I can run the project"
- to "I can explain the project clearly and confidently"

## Start Here

If you want the shortest full picture:

1. `README.md`
2. `docs/project_map.md`
3. `docs/interview_notes.md`

That gives you the public story, the technical flow, and the interview version.

## Best Reading Paths

### Path 1: Understand the ML pipeline

Read these in order:

1. `docs/project_map.md`
2. `docs/preprocessing_cheatsheet.md`
3. `docs/feature_variant_cheatsheet.md`
4. `docs/ml_concepts_for_this_project.md`
5. `docs/metrics_and_model_reading_guide.md`

Use this path if your goal is:

- understanding EO and EC
- explaining Welch PSD bandpower features
- understanding the models and metrics
- explaining why the benchmark is reproducible

### Path 2: Understand the code flow

Read these in order:

1. `docs/file_by_file_code_map.md`
2. `src/train_lemon_multitarget.py`
3. `src/feature_variants.py`
4. `src/split_manifest.py`
5. `src/benchmark_v1.py`
6. `run_resume_benchmark.py`

Use this path if your goal is:

- understanding where each step happens in code
- following data flow from raw EEG to saved results
- explaining the benchmark scripts in interviews

### Path 3: Understand the results

Read these in order:

1. `docs/results_interpretation.md`
2. `docs/metrics_and_model_reading_guide.md`
3. `results/benchmark_v1/<run_id>/summary_results.csv`

Use this path if your goal is:

- describing the strongest targets honestly
- explaining why `random_forest` beat `ridge`
- keeping the results constructive and benchmark-first

### Path 4: Understand the web demo layer

Read these in order:

1. `docs/web_layer_for_interviews.md`
2. `backend/app/main.py`
3. `backend/app/services/predictor.py`
4. `backend/app/services/feature_builder.py`
5. `frontend/app/HomeClient.tsx`

Use this path if your goal is:

- explaining the lightweight FastAPI + Next.js demo
- understanding anchor inputs and feature expansion
- keeping the web layer secondary to the benchmark

## Interview Prep Set

If you only review four docs before an interview, review:

1. `docs/project_map.md`
2. `docs/interview_notes.md`
3. `docs/results_interpretation.md`
4. `docs/web_layer_for_interviews.md`

## What Each Doc Is For

- `project_map.md`
  repo-wide beginner-friendly walkthrough
- `preprocessing_cheatsheet.md`
  the shortest correct explanation of the EEG preprocessing and bandpower pipeline
- `feature_variant_cheatsheet.md`
  what the feature variants mean and which ones matter most
- `ml_concepts_for_this_project.md`
  the ML ideas that matter specifically for this repo
- `metrics_and_model_reading_guide.md`
  how to read `summary_results.csv` and describe results correctly
- `results_interpretation.md`
  benchmark-first interpretation of the saved runs
- `file_by_file_code_map.md`
  concise explanation of the most important files
- `interview_notes.md`
  ready-to-practice answers and project explanations
- `web_layer_for_interviews.md`
  the repo-specific backend/frontend concepts that matter

## A Good Study Sequence

One practical sequence is:

1. read `README.md`
2. read `docs/project_map.md`
3. skim the code with `docs/file_by_file_code_map.md`
4. study the preprocessing, feature, ML, and metrics docs
5. read `docs/results_interpretation.md`
6. practice with `docs/interview_notes.md`
7. review `docs/web_layer_for_interviews.md` last

That keeps the benchmark-first ML story central and the demo layer secondary.
