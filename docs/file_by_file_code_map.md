# File-By-File Code Map

This is a concise map of the files that matter most.

## `src/train_lemon_multitarget.py`

What this file is for:

- build the base subject-level EEG dataset
- train the saved random-forest model used by the demo paths
- save the feature schema and scaler metadata

Main functions:

- `load_metadata_and_targets()`
- `find_subject_file_pairs()`
- `load_and_preprocess_eeg()`
- `compute_bandpower_features()`
- `build_dataset()`
- `save_outputs()`
- `main()`

Execution flow:

1. load metadata and cognitive targets
2. find subjects with both EO and EC files
3. preprocess EO and EC recordings
4. compute bandpower features
5. build one row per subject
6. standardize targets for training
7. train the final RF model
8. save model and processed artifacts

What an interviewer might ask:

- "How do you turn raw EEG into a training row?"
- "Where do age and gender enter?"
- "What artifacts get saved for later inference?"

Important blocks to understand:

- `load_and_preprocess_eeg()`
- `compute_bandpower_features()`
- `build_dataset()`
- `save_outputs()`
- `main()`

## `src/feature_variants.py`

What this file is for:

- define the public and exploratory feature variants
- transform the base EO/EC table into alternate feature spaces

Main functions:

- `split_eo_ec_columns()`
- `build_regional_mean_features()`
- `build_per_channel_ratio_features()`
- `build_frontal_alpha_asymmetry_features()`
- `build_feature_variant()`

Execution flow:

1. split EO and EC columns
2. confirm safe EO/EC pairing
3. build the requested derived features
4. append context columns like `age` and `gender`

What an interviewer might ask:

- "Why do you have multiple feature variants?"
- "Which variants are public defaults?"
- "How do region features get created?"

Important blocks to understand:

- `FEATURE_VARIANTS`
- `ADVANCED_FEATURE_VARIANTS`
- `infer_region()`
- `build_regional_mean_features()`
- `build_feature_variant()`

## `src/split_manifest.py`

What this file is for:

- create, load, validate, and summarize subject-level fold assignments

Main functions:

- `generate_split_manifest()`
- `load_split_manifest()`
- `validate_split_manifest()`
- `fold_counts()`

Execution flow:

1. deduplicate subject IDs
2. create KFold splits at the subject level
3. write one row per `(fold, id)` with `train` or `test`
4. validate future runs against the dataset subject list

What an interviewer might ask:

- "Why save a split manifest instead of just rerunning KFold?"
- "How do you avoid subject leakage?"

Important blocks to understand:

- `generate_split_manifest()`
- `validate_split_manifest()`
- `fold_counts()`

## `src/benchmark_v1.py`

What this file is for:

- run the reproducible benchmark
- compare models and feature variants on fixed subject-level splits
- write benchmark CSVs and metadata

Main functions:

- `parse_args()`
- `build_subject_level_dataset()`
- `enforce_target_lock()`
- `build_model()`
- `evaluate_models()`
- `summarize_results()`
- `main()`

Execution flow:

1. parse CLI arguments
2. build the subject-level dataset
3. enforce target-column lock
4. load or generate the split manifest
5. build each requested feature variant
6. fit each requested model on each fold
7. inverse-transform predictions to raw target space
8. write `fold_results.csv`, `summary_results.csv`, and `metadata.json`

What an interviewer might ask:

- "How do you keep the default benchmark small and explainable?"
- "How do you prevent target leakage?"
- "What files prove the run is reproducible?"

Important blocks to understand:

- `DEFAULT_MODEL_NAMES`
- `parse_args()`
- `enforce_target_lock()`
- `evaluate_models()`
- `main()`

## `run_resume_benchmark.py`

What this file is for:

- give the repo one obvious interview-friendly benchmark command

Main functions:

- `parse_args()`
- `build_benchmark_argv()`
- `main()`

Execution flow:

1. read a small CLI
2. force the public model list and public feature-variant list
3. delegate to `src.benchmark_v1`

What an interviewer might ask:

- "Why have a separate wrapper script?"

Important blocks to understand:

- `RESUME_MODELS`
- `RESUME_FEATURE_VARIANTS`
- `build_benchmark_argv()`
- `main()`

## `src/predict_my_report.py`

What this file is for:

- demo sparse-input inference from a small set of report-style anchor values

Main functions:

- `load_model()`
- `load_feature_schema()`
- `prepare_state_maps()`
- `build_final_input_dataframe()`
- `main()`

Execution flow:

1. load saved artifacts
2. start from a small set of anchor channel values
3. estimate missing channels using simple region templates
4. build a schema-compatible input row
5. run prediction and print outputs

What an interviewer might ask:

- "How do you go from sparse report values to the full saved feature schema?"
- "Why is this secondary to the benchmark?"

Important blocks to understand:

- `REPORT_FEATURES`
- `select_region_template()`
- `build_final_input_dataframe()`
- `print_exploratory_warning()`

## `backend/app/main.py`

What this file is for:

- define the FastAPI demo routes
- load the prediction service at app startup

Main functions:

- `build_demo_anchor_values()`
- `lifespan()`
- `predict_manual()`
- `predict_from_csv()`
- `predict_from_pdf()`
- `predict_demo()`

Execution flow:

1. app starts
2. `PredictionService` loads saved artifacts once
3. routes validate input and call prediction helpers
4. results are returned as JSON

What an interviewer might ask:

- "How does the backend initialize the model?"
- "Which routes are real prediction paths and which are future extensions?"

Important blocks to understand:

- `lifespan()`
- `get_prediction_service()`
- `predict_from_csv()`
- `predict_from_pdf()`
- `predict_demo()`

## `backend/app/services/feature_builder.py`

What this file is for:

- convert a small anchor-input representation into the full feature vector expected by the saved model

Main class:

- `FeatureBuilder`

Execution flow:

1. loop over every saved feature column
2. fill `age` and `gender` directly
3. fill direct anchor features when possible
4. estimate missing channels by simple region rules
5. return a complete vector in schema order

What an interviewer might ask:

- "How do you handle missing non-anchor channels?"
- "How do you keep the vector aligned with the training schema?"

Important blocks to understand:

- `build_vector()`
- `_parse_conditioned_feature()`
- `_estimate_conditioned_channel_value()`
- `_estimate_plain_channel_value()`
- `_channel_group()`

## `backend/app/services/predictor.py`

What this file is for:

- load artifacts and expose prediction helpers for all API routes

Main class:

- `PredictionService`

Execution flow:

1. load model and feature schema
2. build a `FeatureBuilder`
3. normalize anchor input
4. build feature vectors
5. call the model
6. map outputs to named targets

What an interviewer might ask:

- "Where does artifact loading happen?"
- "How do manual and batch prediction share logic?"

Important blocks to understand:

- `load()`
- `predict_from_manual()`
- `predict_from_anchors()`
- `predict_batch()`
- `_normalize_anchors()`

## `frontend/app/HomeClient.tsx`

What this file is for:

- provide the main demo UI for CSV, PDF, manual, and built-in demo workflows

Main functions and structures:

- `buildDefaultManualInputs()`
- `toNumericManualInputs()`
- `handleCsvSubmit()`
- `handlePdfSubmit()`
- `handleDemoClick()`
- `handleManualSubmit()`

Execution flow:

1. initialize state with `useState`
2. collect user input or file uploads
3. send requests with `fetch`
4. render loading, error, single-result, batch-result, or info states

What an interviewer might ask:

- "How does the UI talk to the API?"
- "Which workflow is the most representative?"
- "Why is manual entry hidden under Advanced?"

Important blocks to understand:

- constants for `CONDITIONS`, `ELECTRODES`, and `BANDS`
- `buildDefaultManualInputs()`
- the four submit handlers
- the workflow cards
- the result rendering section
