# Docs Study Guide

This folder is the main internal learning and interview-prep layer for the repository.

Use `README.md` for the external project story.

Use `docs/` when you want to understand:

- what the benchmark is really doing
- how raw EEG becomes tabular ML features
- why the design choices were made
- how the code flows from file to file
- how to explain the project clearly in ML and SWE interviews

Treat `docs/` as the canonical learning layer. If you notice older note files elsewhere in the repo, use `docs/` first.

## The Core Rule for This Repo

Think of the repository in two layers:

- external-facing layer
  `README.md`
- internal learning layer
  `docs/`

If you want the polished public story, read `README.md`.

If you want the real understanding layer, stay in `docs/`.

## What Is Core vs Optional

### Core benchmark path

These are the files and docs that matter most:

- `src/train_lemon_multitarget.py`
- `src/feature_variants.py`
- `src/split_manifest.py`
- `src/benchmark_v1.py`
- `run_resume_benchmark.py`

This path is the center of the project story:

raw EEG -> bandpower features -> feature variants -> fixed subject-level folds -> model comparison -> saved metrics and run artifacts

### Optional demo path

These are useful, but they are secondary:

- `src/predict_my_report.py`
- `backend/app/`
- `frontend/app/`
- `docs/web_layer_for_interviews.md`

This path is best understood as engineering breadth around saved benchmark artifacts, not as the main evaluated contribution.

## Best Reading Order

### 1. Complete beginners

Read these in order:

1. `README.md`
2. `docs/project_map.md`
3. `docs/mental_model_of_data_flow.md`
4. `docs/preprocessing_cheatsheet.md`
5. `docs/feature_variant_cheatsheet.md`
6. `docs/ml_concepts_for_this_project.md`
7. `docs/metrics_and_model_reading_guide.md`
8. `docs/results_interpretation.md`
9. `docs/interview_notes.md`

Why this order:

- it starts with the big picture
- then shows how one subject becomes one row
- then explains the most important ML ideas in plain English
- then ends with results and interview framing

### 2. ML/SWE interview prep

Read these in order:

1. `README.md`
2. `docs/interview_notes.md`
3. `docs/project_map.md`
4. `docs/results_interpretation.md`
5. `docs/feature_variant_cheatsheet.md`
6. `docs/ml_concepts_for_this_project.md`
7. `docs/code_block_questions.md`
8. `docs/web_layer_for_interviews.md`

Why this order:

- it helps you frame the project quickly
- then gives you defensible answers about design choices, results, leakage, and architecture
- then helps you practice likely follow-up questions

### 3. Deep code understanding

Read these in order:

1. `docs/file_by_file_code_map.md`
2. `docs/deep_walkthrough_train_lemon_multitarget.md`
3. `src/train_lemon_multitarget.py`
4. `docs/deep_walkthrough_benchmark_v1.md`
5. `src/benchmark_v1.py`
6. `src/feature_variants.py`
7. `src/split_manifest.py`
8. `run_resume_benchmark.py`
9. `docs/web_layer_for_interviews.md`

Why this order:

- it starts with the map
- then follows the exact execution order of the two main scripts
- then moves into the optional demo layer last

## If You Only Have One Hour

Read these five:

1. `README.md`
2. `docs/project_map.md`
3. `docs/mental_model_of_data_flow.md`
4. `docs/interview_notes.md`
5. `docs/results_interpretation.md`

That set gives you the project story, the pipeline story, and the result story.

## Doc Map

### Core understanding docs

- `project_map.md`
  Beginner-friendly top-to-bottom explanation of the repo.
- `mental_model_of_data_flow.md`
  One running example that follows raw EEG all the way to saved benchmark outputs.
- `preprocessing_cheatsheet.md`
  Short explanation of average reference, Welch PSD, FFT involvement, and bandpower.
- `feature_variant_cheatsheet.md`
  Explains the public feature variants and why they exist.
- `ml_concepts_for_this_project.md`
  Explains the ML ideas that matter in this repo, including leakage and reproducibility.
- `metrics_and_model_reading_guide.md`
  Explains how to read `summary_results.csv` and talk about metrics honestly.
- `results_interpretation.md`
  Gives the honest benchmark-first interpretation of the saved runs.

### Code understanding docs

- `file_by_file_code_map.md`
  Quick map of the most important files and what each one is for.
- `deep_walkthrough_train_lemon_multitarget.md`
  Exact execution-order walkthrough of the training script.
- `deep_walkthrough_benchmark_v1.md`
  Exact execution-order walkthrough of the benchmark script.

### Interview practice docs

- `interview_notes.md`
  Strong, honest ML/SWE interview framing and short project explanations.
- `code_block_questions.md`
  Self-test questions and short ideal answers for the main files and functions.

### Optional demo-layer doc

- `web_layer_for_interviews.md`
  Repo-specific backend/frontend notes for the lightweight demo layer.

## A Good Way to Study With the Code Open

Try this loop:

1. read one doc section
2. open the matching code file
3. explain it out loud in your own words
4. answer one or two questions from `docs/code_block_questions.md`

That is usually better than trying to memorize polished lines.

## Final Reminder

The main project story is:

- reproducible EEG benchmark
- explainable feature-engineering pipeline
- honest comparison of models and feature variants
- modest but encouraging signal on selected targets

The optional web layer is worth understanding, but only after the benchmark path feels clear.
