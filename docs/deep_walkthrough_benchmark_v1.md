# Deep Walkthrough: `src/benchmark_v1.py`

This doc follows the benchmark script in the exact order it runs.

The goal is to help you explain:

- how the benchmark is organized
- how it stays reproducible
- how it avoids leakage and setup drift

## What This Script Is For

`src/benchmark_v1.py` is the main reproducible comparison runner in the repository.

At this point the script is trying to answer:

Which model and feature variant work best on the same fixed subject-level folds?

## Before `main()`

Before `main()` starts, the script defines:

- default paths for the split manifest and results root
- the default model list
- the default public feature-variant list
- CSV column layouts for the outputs

This step matters because the benchmark should feel stable and easy to rerun.

The design philosophy is:

- small default suite
- optional exploratory extensions
- saved outputs that are easy to compare later

## Step 1: Parse CLI Arguments

The first thing `main()` does is:

- `args = parse_args(argv)`

### What this block is doing

It reads command-line options such as:

- split manifest path
- results root
- experiment name
- selected models
- selected feature variants

### Why this exists

Without this step, the benchmark would not know:

- whether to use the default suite or a custom run
- where to save outputs
- whether to regenerate the split manifest

### What goes in

- CLI flags

### What comes out

- a parsed `args` object

### How it fits into the full pipeline

This is the setup step for the benchmark run.

### Beginner confusion point

Confusion:

"Why does the script support more options if the repo story is supposed to stay small?"

Answer:

Because the public default is intentionally small, but the repo still preserves exploratory extensions for side experiments.

### An interviewer may ask

"What happens when you run the benchmark with no extra flags?"

Good answer:

"It runs the public default suite: dummy, ridge, and random forest across the three public feature variants."

## Step 2: Build The Subject-Level Dataset

Next `main()` calls:

- `build_subject_level_dataset()`

### What this block is doing

It reuses the training-side functions from `src/train_lemon_multitarget.py` to:

- load metadata and targets
- find EO/EC file pairs
- build the subject-level dataset

### Why this exists

This step matters because the benchmark should use the same base dataset logic as the training script.

Without this step, the benchmark and saved-model path could drift apart.

### What goes in

- the raw data already defined through the training-side paths

### What comes out

- `X`
  base feature matrix
- `y_raw`
  raw target matrix
- `subject_ids`
  one ID per row
- `resolved_target_columns`
  the actual source-column mapping

### How it fits into the full pipeline

This is the data-loading step for the benchmark.

### Beginner confusion point

Confusion:

"Why does the benchmark import functions from the training script?"

Answer:

Because the benchmark wants the same raw-data-to-base-table logic, not a separate data pipeline.

### An interviewer may ask

"How do you keep the benchmark dataset aligned with the training dataset?"

Good answer:

"The benchmark directly reuses the base dataset-building functions from the training-side script."

## Step 3: Enforce Target Lock

The next call is:

- `enforce_target_lock(...)`

### What this block is doing

It checks whether the resolved target source columns still match the expected configured columns.

### Why this exists

This step matters because target files can change.

Without this step:

- a benchmark rerun might silently use a different source column
- later result comparisons would be much less trustworthy

### What goes in

- the resolved target-column mapping
- the `--allow-target-fallback` flag

### What comes out

- a target-definition record that is later written to metadata

### How it fits into the full pipeline

This is one of the reproducibility guards.

### Beginner confusion point

Confusion:

"Why is column drift such a big deal?"

Answer:

Because if the meaning of the target changes, the whole comparison changes even if the code stayed the same.

### An interviewer may ask

"How do you protect against silent target-definition drift?"

Good answer:

"The benchmark records the expected and resolved target columns and fails unless the drift is explicitly allowed."

## Step 4: Load Or Create The Split Manifest

Next `main()` decides whether to:

- generate a split manifest
- or load an existing one

The relevant calls are:

- `generate_split_manifest(...)`
- `save_split_manifest(...)`
- `load_split_manifest(...)`
- `validate_split_manifest(...)`

### What this block is doing

It ensures that the benchmark uses fixed subject-level train/test assignments.

### Why this exists

This step matters because later comparisons should not change just because the random split changed.

Without this step:

- fold assignments could drift between runs
- model and feature comparisons would be less fair

### What goes in

- subject IDs
- requested number of folds
- random state
- split manifest path

### What comes out

- a validated manifest with one row per subject per fold

### How it fits into the full pipeline

This is the main fairness and leakage-control block for the benchmark.

### Beginner confusion point

Confusion:

"Why save every subject's role in every fold instead of only saving test IDs?"

Answer:

Because the benchmark wants a full reusable partition table that can be validated later.

### An interviewer may ask

"Why save folds to disk instead of relying on `KFold` each time?"

Good answer:

"Because fixed subject-level splits make comparisons repeatable and easier to trust."

## Step 5: Print Dataset And Fold Summary

Before the heavy evaluation starts, `main()` prints:

- the resolved target columns
- the number of dataset subjects
- the per-fold train/test counts

### What this block is doing

It gives the run a quick human-readable check.

### Why this exists

This step matters because you want to see:

- which targets are being used
- how many subjects entered the benchmark
- whether the folds look reasonable

### What goes in

- validated manifest
- target-lock info

### What comes out

- terminal logs

### How it fits into the full pipeline

This is a sanity-check step before model evaluation.

### Beginner confusion point

Confusion:

"Does this step change any data?"

Answer:

No. It is just reporting.

### An interviewer may ask

"What quick evidence do you have that the benchmark run is set up correctly?"

Good answer:

"The script prints resolved targets, subject counts, and fold sizes before training starts."

## Step 6: Create The Output Folder And Skeleton Files

Next `main()` does:

- `init_run_outputs(...)`
- `write_csv_skeleton(...)`
- `write_fold_summary(...)`

### What this block is doing

It creates a timestamped run directory and prepares:

- `fold_results.csv`
- `summary_results.csv`
- `split_summary.csv`
- `metadata.json`

### Why this exists

This step matters because each run should be self-contained and easy to inspect later.

### What goes in

- results root
- experiment name
- fold counts

### What comes out

- one new results directory
- empty or prefilled output files

### How it fits into the full pipeline

This is the file-organization block for reproducibility.

### Beginner confusion point

Confusion:

"Why write empty CSV skeletons before the results exist?"

Answer:

Because the run structure itself is part of the reproducible contract. The files and columns are prepared up front.

### An interviewer may ask

"What files prove the benchmark was reproducible?"

Good answer:

"The run directory stores metadata, fold-level metrics, summary metrics, and split summaries in a fixed layout."

## Step 7: Decide Whether This Is The Default Suite Or A Custom Run

Next `main()` compares the selected models and feature variants against the defaults.

### What this block is doing

It prints whether the run is:

- the small public default suite
- or a custom exploratory selection

### Why this exists

This step matters because the repo wants a clear main story without deleting exploratory flexibility.

### What goes in

- selected model names
- selected feature-variant names

### What comes out

- a clear run description in the terminal

### How it fits into the full pipeline

This is the storytelling block. It keeps the public benchmark path small and readable.

### Beginner confusion point

Confusion:

"If the code supports extra models and variants, why not make them default too?"

Answer:

Because the goal is not to look exhaustive. The goal is to keep the main benchmark easy to understand and defend.

### An interviewer may ask

"How did you keep the benchmark from turning into a confusing kitchen sink?"

Good answer:

"I kept a small public default suite and treated the extra models and variants as opt-in exploratory extensions."

## Step 8: Run `evaluate_models(...)`

This is the main work block of the benchmark.

At this point the script is trying to compare model and feature choices as fairly as possible.

## Step 8A: Loop Over Feature Variants

### What this block is doing

For each requested feature variant, it calls:

- `build_feature_variant(...)`

### Why this exists

Because the benchmark is not just about models. It is also about comparing feature designs.

### What goes in

- the base feature matrix
- one variant name

### What comes out

- one transformed feature matrix for that variant

### Beginner confusion point

Confusion:

"Is the benchmark rebuilding the raw EEG features each time?"

Answer:

No. It starts from the same base subject-level table and then derives feature variants from it.

### An interviewer may ask

"What exactly changes between feature variants?"

Good answer:

"The raw subject-level base table stays the same, but the feature representation changes by adding or removing derived views such as region means or EO-minus-EC differences."

## Step 8B: Loop Over Folds

### What this block is doing

For each fold in the manifest, the script selects:

- train subject IDs
- test subject IDs

Then it slices:

- `X_train`
- `X_test`
- `y_train_raw`
- `y_test_raw`

### Why this exists

This is the held-out evaluation step.

Without this step, the benchmark would not measure generalization on unseen subjects.

### What goes in

- one feature matrix
- one fold from the manifest

### What comes out

- one train/test split for that fold

### Beginner confusion point

Confusion:

"Why are the indices reset earlier and then reattached to subject IDs here?"

Answer:

Because the benchmark wants to align rows with subject IDs for safe fold selection.

### An interviewer may ask

"How do you make sure the benchmark splits at the subject level?"

Good answer:

"The manifest stores subject IDs, and the feature and target tables are indexed by subject ID before slicing."

## Step 8C: Scale Targets On The Train Fold Only

Inside each fold, the script calls:

- `zscore_targets_train_only(y_train_raw)`

### What this block is doing

It computes target mean and standard deviation using only the training fold.

### Why this exists

This step matters because the test fold should be treated like future unseen data.

Without this step:

- test-fold target statistics would leak into training
- the evaluation could look slightly better than it really is

### What goes in

- raw training targets

### What comes out

- scaled training targets
- per-target means
- per-target standard deviations

### How it fits into the full pipeline

This is one of the main leakage-control steps.

### Beginner confusion point

Confusion:

"Why scale only the targets here and not the features for every model?"

Answer:

Because the target-scaling logic is benchmark-wide, while feature scaling is model-dependent. Ridge uses its own feature-scaling pipeline, but random forest does not need it.

### An interviewer may ask

"How do you avoid target leakage?"

Good answer:

"Target scaling is fit on the train fold only, and predictions are later inverse-transformed back to raw target space."

## Step 8D: Loop Over Models

Inside each fold, the script loops through the selected models and calls:

- `build_model(model_name, random_state)`
- `estimator.fit(X_train, y_train_scaled)`
- `estimator.predict(X_test)`

### What this block is doing

It trains the requested model on the current fold and feature variant.

### Why this exists

This is the actual model-comparison step.

### What goes in

- train features
- scaled train targets
- model choice

### What comes out

- scaled predictions for the test fold

### Beginner confusion point

Confusion:

"Why does `build_model()` return different shapes of estimator objects?"

Answer:

Because some models can handle multi-output directly and others are wrapped in `MultiOutputRegressor`.

### An interviewer may ask

"Why include both ridge and random forest?"

Good answer:

"Ridge is the clean linear baseline, and random forest is the stronger nonlinear tabular baseline."

## Step 8E: Convert Predictions Back To Raw Space And Score Them

After prediction, the script does:

- `inverse_scale_predictions(...)`
- `r2_score(...)`
- `mean_squared_error(...)`
- `mean_absolute_error(...)`
- `safe_pearson(...)`

### What this block is doing

It converts scaled predictions back into raw target units and computes fold-level metrics.

### Why this exists

This step matters because the benchmark wants final numbers that are easy to interpret and compare.

Without this step:

- you would only have scaled-space results
- the saved metrics would be less meaningful

### What goes in

- scaled predictions
- target order
- train-fold means and standard deviations
- raw test targets

### What comes out

- one result row per target, per model, per feature variant, per fold

### Beginner confusion point

Confusion:

"Why are metrics computed target by target?"

Answer:

Because each target has its own values, scale, and difficulty. The benchmark reports them separately on purpose.

### An interviewer may ask

"What metrics do you report and in what space?"

Good answer:

"R2, RMSE, MAE, and Pearson correlation, all in raw target space after inverse-transforming predictions."

## Step 9: Summarize The Fold Results

After `evaluate_models(...)` returns, `main()` calls:

- `summarize_results(fold_results)`

### What this block is doing

It groups fold-level rows and computes means and standard deviations across folds.

### Why this exists

This step matters because the benchmark needs one summary table that is easy to read and compare.

### What goes in

- `fold_results`

### What comes out

- `summary_results`

### How it fits into the full pipeline

This is the compression step from detailed fold rows to headline comparison rows.

### Beginner confusion point

Confusion:

"Why keep both fold results and summary results?"

Answer:

Because summary rows are easier to read, but fold rows are still useful for auditing stability and per-fold variation.

### An interviewer may ask

"Why save both fold-level and summary-level metrics?"

Good answer:

"Because summary rows are good for the headline comparison, and fold rows keep the run auditable."

## Step 10: Write Metadata

Near the end, `main()` builds a metadata payload and calls:

- `write_metadata(metadata_path, metadata)`

### What this block is doing

It records:

- timestamp
- run ID
- split-manifest path
- selected models
- selected feature variants
- target definition
- target scaling method

### Why this exists

This step matters because the run should be explainable later without guessing.

### What goes in

- run configuration
- target-lock information
- split information
- model registry info

### What comes out

- `metadata.json`

### How it fits into the full pipeline

This is the reproducibility record for the run.

### Beginner confusion point

Confusion:

"Why save so much metadata if the CSV results already exist?"

Answer:

Because the results alone do not tell you the full setup. The metadata explains how the run was configured.

### An interviewer may ask

"What makes a run easy to audit later?"

Good answer:

"The benchmark saves not just metrics, but also split details, target definitions, selected models and variants, and scaling information."

## Step 11: Print A Human-Readable Summary

At the very end, the script calls:

- `print_human_readable_summary(summary_results)`

### What this block is doing

It prints:

- the best overall row
- the best row per target

### Why this exists

This step makes the benchmark easier to discuss right after it finishes.

### What goes in

- `summary_results`

### What comes out

- readable terminal output

### Beginner confusion point

Confusion:

"Is this printed summary the official saved result?"

Answer:

No. It is just a quick view. The official saved outputs are the CSVs and metadata file.

### An interviewer may ask

"How do you quickly explain a finished run?"

Good answer:

"I look at the best row per target, then use the saved summary CSV for the full comparison."

## Final Mental Picture

You can think of `src/benchmark_v1.py` as:

"The script that freezes the evaluation setup, compares models and feature variants fairly, and writes everything needed to explain the run later."
