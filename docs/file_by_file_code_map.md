# File-By-File Code Map

This doc is for the moment when you open the repo and wonder:

Which files are core, which are optional, and what should I read first?

## Best Way To Use This Doc

1. read this file once
2. open the matching code file
3. use the related deep walkthrough when you want more detail

The two most important walkthroughs are:

- `docs/deep_walkthrough_train_lemon_multitarget.md`
- `docs/deep_walkthrough_benchmark_v1.md`

## Core Benchmark Files

### `run_resume_benchmark.py`

What this file is for:

- the simplest public entrypoint for the benchmark

Main functions:

- `parse_args()`
- `build_benchmark_argv()`
- `main()`

Execution flow:

1. read a small CLI
2. force the public model list and public feature-variant list
3. delegate to `src/benchmark_v1.py`

What an interviewer may ask:

- "Why have a wrapper script at all?"

Most important blocks:

- `RESUME_MODELS`
- `RESUME_FEATURE_VARIANTS`
- `build_benchmark_argv()`

Related doc:

- `docs/deep_walkthrough_benchmark_v1.md`

### `src/benchmark_v1.py`

What this file is for:

- the main reproducible benchmark runner

Main functions:

- `parse_args()`
- `build_subject_level_dataset()`
- `enforce_target_lock()`
- `evaluate_models()`
- `summarize_results()`
- `main()`

Execution flow:

1. parse CLI arguments
2. build the subject-level dataset
3. check target-column definitions
4. load or create the split manifest
5. build each feature variant
6. train each model on each fold
7. convert predictions back to raw target space
8. write summary CSVs and metadata

What an interviewer may ask:

- "How do you keep the benchmark reproducible?"

Most important blocks:

- `DEFAULT_MODEL_NAMES`
- `parse_args()`
- `enforce_target_lock()`
- `zscore_targets_train_only()`
- `evaluate_models()`
- `main()`

Related doc:

- `docs/deep_walkthrough_benchmark_v1.md`

### `src/train_lemon_multitarget.py`

What this file is for:

- build the base subject-level dataset from raw EEG and tabular metadata
- train the saved random-forest model used by the optional demo paths

Main functions:

- `load_metadata_and_targets()`
- `find_subject_file_pairs()`
- `load_and_preprocess_eeg()`
- `compute_bandpower_features()`
- `build_dataset()`
- `standardize_targets()`
- `save_outputs()`
- `main()`

Execution flow:

1. load metadata and targets
2. find subjects with both EO and EC files
3. preprocess EEG
4. compute bandpower features
5. build one row per subject
6. standardize targets
7. run a quick CV evaluation
8. fit the final model on all data
9. save artifacts

What an interviewer may ask:

- "How do you turn raw EEG into one tabular row?"

Most important blocks:

- `load_metadata_and_targets()`
- `find_subject_file_pairs()`
- `load_and_preprocess_eeg()`
- `compute_bandpower_features()`
- `build_dataset()`
- `main()`

Related doc:

- `docs/deep_walkthrough_train_lemon_multitarget.md`

### `src/feature_variants.py`

What this file is for:

- define the public and exploratory feature variants

Main functions:

- `split_eo_ec_columns()`
- `build_regional_mean_features()`
- `build_per_channel_ratio_features()`
- `build_frontal_alpha_asymmetry_features()`
- `build_feature_variant()`

Execution flow:

1. separate EO and EC columns
2. verify safe EO/EC pairing
3. build any requested derived features
4. append context columns such as `age` and `gender`

Note:

The EO-minus-EC difference and EO/EC log-ratio logic live inside `build_feature_variant()` rather than in separate top-level helper functions.

What an interviewer may ask:

- "Why do you have multiple feature variants instead of one final feature table?"

Most important blocks:

- `FEATURE_VARIANTS`
- `ADVANCED_FEATURE_VARIANTS`
- `FEATURE_VARIANT_DESCRIPTIONS`
- `infer_region()`
- `build_feature_variant()`

Related docs:

- `docs/feature_variant_cheatsheet.md`
- `docs/mental_model_of_data_flow.md`

### `src/split_manifest.py`

What this file is for:

- create, save, load, and validate fixed subject-level folds

Main functions:

- `generate_split_manifest()`
- `save_split_manifest()`
- `load_split_manifest()`
- `validate_split_manifest()`
- `fold_counts()`

Execution flow:

1. deduplicate subject IDs
2. create KFold assignments at the subject level
3. write one row per `(fold, id, split)`
4. validate later runs against the current dataset IDs

What an interviewer may ask:

- "Why save folds instead of generating them fresh every time?"

Most important blocks:

- `generate_split_manifest()`
- `validate_split_manifest()`
- `fold_counts()`

Related docs:

- `docs/ml_concepts_for_this_project.md`
- `docs/deep_walkthrough_benchmark_v1.md`

## Optional Demo Files

### `src/predict_my_report.py`

What this file is for:

- exploratory sparse-input inference from a small set of report-style anchor values

Main functions:

- `print_demo_extension_note()`
- `select_region_template()`
- `build_final_input_dataframe()`
- `main()`

Execution flow:

1. load saved artifacts
2. start from a small set of anchor values
3. estimate missing channels with simple region templates
4. build a schema-compatible row
5. run prediction and print outputs

What an interviewer may ask:

- "Why is this secondary to the benchmark?"

Most important blocks:

- `DEMO_REPORT_FEATURES`
- `select_region_template()`
- `build_final_input_dataframe()`
- `print_demo_extension_note()`

### `backend/app/main.py`

What this file is for:

- define the FastAPI demo routes
- load the prediction service at app startup

Main functions:

- `lifespan()`
- `get_prediction_service()`
- `predict_manual()`
- `predict_from_csv()`
- `predict_from_pdf()`
- `predict_demo()`

Execution flow:

1. app starts
2. `PredictionService` loads saved artifacts once
3. routes validate input and call prediction helpers
4. results are returned as JSON

What an interviewer may ask:

- "Which API routes are real prediction paths and which are future extensions?"

Most important blocks:

- `lifespan()`
- `predict_from_csv()`
- `predict_from_pdf()`
- `predict_demo()`

Related doc:

- `docs/web_layer_for_interviews.md`

### `backend/app/services/feature_builder.py`

What this file is for:

- expand a small anchor-input representation into the full feature vector expected by the saved model

Main class:

- `FeatureBuilder`

Execution flow:

1. loop over every saved feature column
2. fill `age` and `gender`
3. fill direct anchor features when available
4. estimate missing channels with simple region rules
5. return a complete vector in schema order

What an interviewer may ask:

- "How do you go from a small manual input to the full saved schema?"

Most important blocks:

- `build_vector()`
- `_estimate_conditioned_channel_value()`
- `_estimate_plain_channel_value()`
- `_channel_group()`

Related doc:

- `docs/web_layer_for_interviews.md`

### `backend/app/services/predictor.py`

What this file is for:

- load saved artifacts and expose shared prediction helpers for all routes

Main class:

- `PredictionService`

Execution flow:

1. load model and feature schema
2. create `FeatureBuilder`
3. normalize anchor input
4. build feature vectors
5. call the saved model
6. map outputs to named targets

What an interviewer may ask:

- "How do manual and batch prediction share logic?"

Most important blocks:

- `load()`
- `predict_from_manual()`
- `predict_from_anchors()`
- `predict_batch()`

Related doc:

- `docs/web_layer_for_interviews.md`

### `frontend/app/HomeClient.tsx`

What this file is for:

- provide the main demo UI for CSV, PDF, manual, and built-in demo workflows

Main structures and handlers:

- `buildDefaultManualInputs()`
- `handleCsvSubmit()`
- `handlePdfSubmit()`
- `handleDemoClick()`
- `handleManualSubmit()`

Execution flow:

1. initialize UI state
2. collect user input or file uploads
3. send requests with `fetch`
4. render loading, errors, predictions, or info messages

What an interviewer may ask:

- "Which UI workflow is the most representative?"

Most important blocks:

- workflow constants
- the submit handlers
- `ResultsPanel`
- the CSV workflow card

Related doc:

- `docs/web_layer_for_interviews.md`
