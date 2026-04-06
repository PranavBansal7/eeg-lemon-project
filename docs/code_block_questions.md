# Code Block Questions

Use this file to test whether you really understand the code.

The goal is not to memorize exact wording.

The goal is to make sure you can explain:

- what each main file does
- why the main blocks exist
- what the design choices were

## `run_resume_benchmark.py`

### Very easy recall

Q: What is this file for?

A: It gives the repo one obvious command for the public default benchmark run.

### Understanding

Q: What does this wrapper actually change compared with `src/benchmark_v1.py`?

A: It narrows the CLI and forces the public model list and public feature-variant list before delegating to `src/benchmark_v1.py`.

### Design choice

Q: Why keep a separate wrapper instead of telling users to always call the full benchmark script?

A: Because the repo wants one simple, benchmark-first entrypoint that is easy to run and explain.

## `src/train_lemon_multitarget.py`

### Very easy recall

Q: What are the four main stages of this script?

A: Load metadata and targets, load and preprocess EEG, compute bandpower features, then train/evaluate/save the baseline artifacts.

### Understanding

Q: What is the main output of this script besides the saved model?

A: A subject-level feature table plus the feature schema and target-scaling metadata.

### Design choice

Q: Why does this script require both EO and EC files for a subject?

A: Because the project story is built around paired EO/EC feature engineering and later EO/EC feature comparisons.

### Short ideal interview answer

"This script is the raw-data-to-artifacts pipeline. It builds the base subject-level dataset and saves the reusable training artifacts."

## `load_metadata_and_targets()`

### Very easy recall

Q: What does this function merge together?

A: Metadata plus four target tables.

### Understanding

Q: Why does it return both a merged table and a target-column mapping?

A: Because the repo wants both the usable data and a record of which source columns were actually used.

### Design choice

Q: Why not hard-code the target columns and assume they always exist?

A: Because phenotype files can drift, and the benchmark wants to detect or log that instead of failing silently.

## `find_subject_file_pairs()`

### Very easy recall

Q: What does this function look for?

A: Matching `*_EO.set` and `*_EC.set` files by subject ID.

### Understanding

Q: What does it return?

A: A mapping from subject ID to EO and EC file paths.

### Design choice

Q: Why is this matching step separate from feature extraction?

A: Because subject pairing is a data-availability problem, and keeping it separate makes the pipeline easier to explain.

## `load_and_preprocess_eeg()` and `compute_bandpower_features()`

### Very easy recall

Q: What preprocessing happens before feature extraction?

A: EEG-channel selection, bad-channel exclusion, and average reference.

### Understanding

Q: What does `compute_bandpower_features()` actually output?

A: A dictionary of feature names like `f3_alpha_eo` mapped to bandpower values.

### Design choice

Q: Why use Welch PSD bandpower instead of feeding the raw time series directly into a model?

A: Because the project is centered on a clear, explainable tabular benchmark, and bandpower is a readable spectral summary.

### Short ideal interview answer

"The code turns each recording into channel-by-band spectral summaries, which is much easier to benchmark and explain than raw EEG sequences."

## `build_dataset()`

### Very easy recall

Q: What is the prediction unit in this repo?

A: The subject.

### Understanding

Q: What does one row in `X` contain?

A: EO features, EC features, age, and gender.

### Design choice

Q: Why append age and gender to the same table as the EEG features?

A: Because the downstream models are tabular regressors and expect one aligned feature matrix.

## `src/feature_variants.py`

### Very easy recall

Q: What are the three public feature variants?

A: `eo_ec_concat`, `eo_ec_concat_plus_regions`, and `eo_ec_concat_plus_diff_plus_regions`.

### Understanding

Q: What problem are feature variants solving?

A: They let the benchmark compare feature designs instead of pretending one final feature table is obviously correct.

### Design choice

Q: Why are the more complex ratio and asymmetry variants not the main story?

A: Because the simpler variants are easier to explain and they remain the stronger public benchmark story in the saved runs.

### Short ideal interview answer

"I built a small feature-ablation ladder so I could compare EO/EC handling, regional summaries, and explicit EO-minus-EC contrast in a reproducible way."

## `src/split_manifest.py`

### Very easy recall

Q: What is a split manifest?

A: A saved table of subject-level fold assignments.

### Understanding

Q: Why is the manifest keyed by subject ID instead of row number?

A: Because the benchmark wants subject-level splitting and later validation against the current dataset IDs.

### Design choice

Q: Why save split manifests to disk at all?

A: Because fixed folds make later comparisons fairer and easier to trust.

## `src/benchmark_v1.py`

### Very easy recall

Q: What is the main purpose of this script?

A: Run the reproducible benchmark and compare models and feature variants on fixed subject-level folds.

### Understanding

Q: What are the main saved outputs of this script?

A: `metadata.json`, `split_summary.csv`, `fold_results.csv`, and `summary_results.csv`.

### Design choice

Q: Why does this script import dataset-building functions from `src/train_lemon_multitarget.py`?

A: Because the benchmark should reuse the same base data logic instead of maintaining a second independent pipeline.

### Short ideal interview answer

"This is the main comparison runner. It freezes the evaluation setup, runs the model and feature comparisons, and writes everything needed to audit the result later."

## `enforce_target_lock()` and `zscore_targets_train_only()`

### Very easy recall

Q: What does target lock protect against?

A: Silent target-definition drift.

### Understanding

Q: Why is target scaling fit on the train fold only?

A: To avoid leaking test-fold target statistics into training.

### Design choice

Q: Why compute metrics in raw target space after inverse-transforming predictions?

A: Because raw-space metrics are easier to interpret and compare honestly.

## `evaluate_models()`

### Very easy recall

Q: What are the three loops inside `evaluate_models()`?

A: Feature-variant loop, fold loop, and model loop.

### Understanding

Q: What does one result row represent?

A: One target for one model, one feature variant, and one fold.

### Design choice

Q: Why build the feature variant once per variant and then reuse it across folds?

A: Because the feature representation is part of the comparison setup, and the fold logic then evaluates it fairly on fixed partitions.

## `docs/results_interpretation.md`

### Very easy recall

Q: What is the cleanest main result story?

A: Modest but encouraging signal on attention and executive function, with more limited signal on working memory and intelligence.

### Understanding

Q: Why is `random_forest` stronger than `ridge` here?

A: The current feature space likely contains nonlinear interactions, which random forest handles more naturally.

### Design choice

Q: Why not oversell the results?

A: Because the value of the repo is the reproducible benchmark and comparison design, not a claim of broad high-accuracy cognitive prediction.

## Optional Demo Layer

### `backend/app/main.py`

Q: What is the backend for?

A: Serving saved benchmark artifacts through manual, CSV, demo, and future-extension PDF routes.

Q: What is the main safe framing?

A: It is a lightweight demo layer around saved artifacts, not the main validated contribution.

### `backend/app/services/feature_builder.py`

Q: What is the hardest demo-layer idea to explain?

A: The backend starts from a small set of anchor EEG values and expands them into the full saved feature schema.

### `frontend/app/HomeClient.tsx`

Q: Which workflow is the best one to discuss first?

A: The CSV workflow, because it is the clearest batch demo path.

## Final Self-Test

If you can answer these four quickly, you are in good shape:

1. How does one subject become one row?
2. Why are EO and EC kept separate?
3. Why do split manifests and train-fold-only target scaling matter?
4. Why is this strongest as a benchmark-first project rather than a prediction product?
