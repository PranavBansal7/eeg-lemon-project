# Deep Walkthrough: `src/train_lemon_multitarget.py`

This doc follows the training script in the exact order it runs.

The goal is to help you answer two questions:

- What is this script doing?
- Why does each step exist?

## What This Script Is For

`src/train_lemon_multitarget.py` does the base data-building job for the repository.

You can think of it as the script that turns raw EEG plus tabular metadata into:

- a subject-level feature table
- a saved random-forest model
- saved artifact files that later code can reuse

## Before `main()`

Before the script even starts running `main()`, it defines:

- project paths such as `DATA_DIR`, `EEG_DIR`, and `MODEL_PATH`
- the target CSV paths
- the band definitions
- the target aliases

At this point the script is trying to make the rest of the pipeline explicit and reproducible.

This step matters because later functions need one clear source of truth for:

- where the data lives
- which target columns to prefer
- which EEG bands to compute

## Step 1: Load Metadata And Targets

The first call in `main()` is:

- `load_metadata_and_targets()`

### What this block is doing

It loads:

- subject metadata
- working memory target table
- attention target table
- executive function target table
- intelligence target table

Then it merges them on subject ID.

### Why this exists

Without this step, the script would not know:

- which subjects have usable age and gender
- which subjects have all target values
- which exact source columns ended up feeding the benchmark

### What goes in

- metadata CSV path
- one CSV path per target
- preferred target column names such as `TAP_WM_1`

### What comes out

- one merged table with `ID`, `age`, `gender`, and the four targets
- a mapping that records which source column was used for each target

### How it fits into the full pipeline

This is the tabular supervision side of the project.

The EEG files alone are not enough. The model also needs:

- subject metadata
- target values to predict

### Beginner confusion point

Confusion:

"Why is the script so careful about choosing target columns?"

Answer:

Because phenotype files can change over time. The code prefers a named target column, but it also logs what happened so the benchmark does not silently drift.

### An interviewer may ask

"How do you know the target values are coming from the intended source columns?"

Good answer:

"The script records the resolved source columns and the benchmark later checks them again through target locking."

## Step 2: Find Subjects With Both EO And EC Files

The next call in `main()` is:

- `find_subject_file_pairs()`

### What this block is doing

It scans the EEG folder for:

- `*_EO.set`
- `*_EC.set`

Then it keeps only subjects who have both files.

### Why this exists

This step matters because the project story depends on EO and EC being compared or combined in a controlled way.

Without this step:

- some subjects would have only EO
- some would have only EC
- the EO/EC feature story would break

### What goes in

- the EEG directory path

### What comes out

- a dictionary like:
  `subject_id -> {"eo": eo_path, "ec": ec_path}`

### How it fits into the full pipeline

This is the gate that decides which raw EEG recordings can even enter the project.

### Beginner confusion point

Confusion:

"Why not use subjects who only have EO?"

Answer:

Because the main benchmark is built around paired EO and EC recordings. Using one condition only would change the feature story and make comparisons less consistent.

### An interviewer may ask

"Why require both EO and EC for a usable subject?"

Good answer:

"Because EO-versus-EC structure is one of the main domain-aware choices in the feature design."

## Step 3: Build The Subject-Level Dataset

The next call in `main()` is:

- `build_dataset(metadata_targets, subject_file_pairs)`

At this point the script is trying to turn many raw files into one ML table.

## Step 3A: Loop Over Subjects

### What this block is doing

For each subject with paired EEG files and a merged metadata row, the script:

1. loads EO EEG
2. loads EC EEG
3. computes EO features
4. computes EC features
5. appends age and gender
6. stores the four targets

### Why this exists

This is the block that creates the final training examples.

One subject becomes:

- one feature row
- one target row

### What goes in

- the merged metadata/targets table
- the paired EO/EC file paths

### What comes out

- `X`
  the feature matrix
- `y_raw`
  the raw target matrix
- `features_df`
  a combined table that also keeps IDs and targets for export

### How it fits into the full pipeline

This is the bridge between raw input files and tabular ML.

### Beginner confusion point

Confusion:

"Why does the function return three different tables?"

Answer:

Because later parts of the repo need slightly different views:

- `X` for model input
- `y_raw` for target values
- `features_df` for saving a more complete artifact table

### An interviewer may ask

"What is the prediction unit in this repo?"

Good answer:

"The prediction unit is the subject. One subject becomes one row."

## Step 3B: Load And Preprocess EEG

Inside the subject loop, the script calls:

- `load_and_preprocess_eeg(eo_path)`
- `load_and_preprocess_eeg(ec_path)`

### What this block is doing

It reads one EEGLAB file with MNE, then:

- keeps EEG channels only
- drops marked bad channels
- applies average reference

### Why this exists

This step matters because the spectral features should be computed from a clean and consistent EEG view.

Without this step:

- non-EEG channels might stay in the data
- bad channels might distort features
- the reference choice would be inconsistent

### What goes in

- one `.set` EEG file

### What comes out

- one preprocessed MNE `Raw` object

### How it fits into the full pipeline

This is the preparation step before spectral feature extraction.

### Beginner confusion point

Confusion:

"Does this script do a huge amount of EEG preprocessing?"

Answer:

No. It keeps preprocessing intentionally light. The project is not trying to be a full EEG preprocessing platform.

### An interviewer may ask

"What preprocessing choices did you make?"

Good answer:

"I kept EEG channels, used marked bad channels, applied average reference, and then extracted spectral bandpower features."

## Step 3C: Compute Bandpower Features

Still inside the subject loop, the script calls:

- `compute_bandpower_features(raw_eo, "eo")`
- `compute_bandpower_features(raw_ec, "ec")`

### What this block is doing

It gets the EEG matrix from MNE and then:

1. computes Welch PSD from 1 to 30 Hz
2. loops over channels
3. loops over named bands
4. integrates PSD values inside each band
5. stores features like `f3_alpha_eo`

### Why this exists

This is the core feature-engineering step.

You can think of it as:

"Turn long EEG recordings into readable channel-by-band summary features."

### What goes in

- one preprocessed EEG recording
- one state suffix such as `eo` or `ec`

### What comes out

- a dictionary of feature names to bandpower values

### How it fits into the full pipeline

This is where raw time-series data becomes tabular ML data.

### Beginner confusion point

Confusion:

"Why does the function use both channel names and state suffixes in feature names?"

Answer:

Because the final table needs to remember:

- which channel the value came from
- which band it came from
- whether it came from EO or EC

### An interviewer may ask

"Can you explain one feature column?"

Good answer:

"`f3_alpha_eo` means alpha-band power from the F3 channel in the eyes-open recording."

## Step 3D: Combine Features, Metadata, And Targets

### What this block is doing

After EO and EC features are ready, the script creates one combined row:

- all EO features
- all EC features
- `age`
- `gender`

It also creates one target row:

- `working_memory`
- `attention`
- `executive_function`
- `intelligence`

### Why this exists

This step matters because the model needs one aligned input row and one aligned target row per subject.

### What goes in

- EO feature dictionary
- EC feature dictionary
- metadata row
- target values

### What comes out

- one complete subject example

### How it fits into the full pipeline

This is the exact moment where "one subject becomes one row" happens.

### Beginner confusion point

Confusion:

"Why are age and gender added directly into the same table as EEG features?"

Answer:

Because the model is a tabular regressor. All input features, whether EEG-derived or metadata-derived, need to live in the same feature matrix.

### An interviewer may ask

"Where do age and gender enter the pipeline?"

Good answer:

"They are appended as simple metadata features during subject-level row construction."

## Step 4: Standardize Targets

Back in `main()`, after the dataset is built, the script checks:

- `STANDARDIZE_TARGETS`

If true, it calls:

- `standardize_targets(y_raw)`

### What this block is doing

It computes the mean and standard deviation for each target and converts targets into z-scores.

### Why this exists

This step matters because the four targets can live on different scales.

Target standardization can make multi-target training more stable.

### What goes in

- the raw target table

### What comes out

- `y_train`
  the scaled target table
- target means
- target standard deviations

### How it fits into the full pipeline

The model trains on scaled targets, but the script still saves the raw-space scaling information for later use.

### Beginner confusion point

Confusion:

"If the targets are scaled, does that mean the final outputs are meaningless?"

Answer:

No. The scaling metadata is saved, so later code can map predictions back to raw target space.

### An interviewer may ask

"Why standardize the targets at all?"

Good answer:

"Because it makes the multi-target optimization problem more numerically stable while still letting me keep raw-space meaning through the saved scaling metadata."

## Step 5: Run A Quick Cross-Validation Evaluation

The next call in `main()` is:

- `evaluate_model(...)`

### What this block is doing

It runs a quick KFold evaluation using the current dataset and prints per-target metrics.

### Why this exists

This step gives the script a sanity-check evaluation before fitting the final saved model on all available data.

### What goes in

- feature matrix
- training targets
- selected target-column mapping

### What comes out

- printed CV metrics in the terminal

### How it fits into the full pipeline

This step is for quick local feedback, not the full benchmark study. The richer comparison work happens later in `src/benchmark_v1.py`.

### Beginner confusion point

Confusion:

"Why does this script both evaluate and then still train on all the data?"

Answer:

Because the quick evaluation is a sanity check, while the final saved model is meant to use all available training data for later demo or inference paths.

### An interviewer may ask

"Is this the main benchmark runner?"

Good answer:

"No. This script builds the base dataset and saved artifacts. The main reproducible comparison runner is `src/benchmark_v1.py`."

## Step 6: Train The Final Model On All Available Data

After the quick evaluation, `main()` does:

- `model = build_model()`
- `model.fit(X, y_train)`

### What this block is doing

It trains the final saved model on the full available dataset.

### Why this exists

This step gives the repo:

- a trained model artifact
- a consistent saved model for the optional demo paths

### What goes in

- the full feature matrix
- the full training target matrix

### What comes out

- one fitted `MultiOutputRegressor(RandomForestRegressor(...))`

### How it fits into the full pipeline

This is the artifact-serving side of the repo, not the main benchmark-comparison side.

### Beginner confusion point

Confusion:

"Why save one final model if the benchmark compares many models?"

Answer:

Because the saved-model path and the benchmark path serve different purposes. The benchmark is for comparison. The saved final model is for the optional demo and artifact-serving workflows.

### An interviewer may ask

"What exact model gets saved here?"

Good answer:

"A multi-output random forest model trained on the full dataset."

## Step 7: Save Outputs

The final call in `main()` is:

- `save_outputs(...)`

### What this block is doing

It writes:

- the trained model
- the full feature table
- the ordered feature schema
- the target scaling metadata

### Why this exists

Without this step:

- later demo routes could not load the model
- the backend would not know the expected feature order
- predictions could not be mapped back cleanly after target scaling

### What goes in

- fitted model
- feature matrix
- combined saved table
- scaling payload

### What comes out

- `models/rf_model.pkl`
- `processed/features.csv`
- `processed/feature_columns.json`
- `processed/target_scaler.json`

### How it fits into the full pipeline

This is the export step that makes the rest of the repo reusable.

### Beginner confusion point

Confusion:

"Why save the feature schema separately if the model is already saved?"

Answer:

Because later inference code must rebuild the input vector in the exact same column order the model saw during training.

### An interviewer may ask

"What artifacts does this script save, and why do they matter?"

Good answer:

"It saves the model, the feature schema, the processed feature table, and target-scaling metadata so later benchmark and demo paths can stay aligned with training."

## Final Mental Picture

At the end of this script, the repo now has:

- one subject-level EEG feature table
- one saved random-forest artifact
- the feature order needed for later inference
- the target-scaling metadata needed to interpret predictions

You can think of `src/train_lemon_multitarget.py` as:

"The script that turns raw EEG and phenotype tables into reusable training artifacts."
