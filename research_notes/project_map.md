# Project Map

## End-to-end flow
1. Load metadata and cognitive targets
2. Find EO and EC EEG files for each subject
3. Load EEG data
4. Compute bandpower features from EEG
5. Build a subject-level dataset
6. Form benchmark feature variants
7. Load or create fixed subject-level CV splits
8. Train benchmark models fold by fold
9. Compute metrics per target
10. Save results and summaries

## Main files

### `src/train_lemon_multitarget.py`
Main training/data assembly core.
Responsible for:
- loading metadata and targets
- finding EO/EC files
- computing EEG bandpower features
- building the base subject-level dataset
- saving training-related outputs

### `src/feature_variants.py`
Builds derived benchmark feature variants from the base EO/EC feature table.
Responsible for:
- EO-only / EC-only views
- concat / diff / log-ratio
- regional summaries
- ratio-based extensions
- asymmetry features
- fail-fast checks for safe pairing

### `src/split_manifest.py`
Creates and validates subject-level split manifests.
Responsible for:
- deterministic fold definitions
- subject-ID-based split persistence
- split validation
- fold count summaries

### `src/benchmark_v1.py`
Benchmark runner.
Responsible for:
- loading the dataset
- loading split manifests
- locking target definitions
- selecting models and feature variants
- fold-wise training/evaluation
- writing metadata and CSV outputs

## Current benchmark best setup
- Model: `random_forest`
- Feature variant: `eo_ec_concat_plus_diff_plus_regions`

## Important methodological choices
- Subject-level fixed CV splits
- Train-fold-only target scaling
- Metrics computed in raw target space
- EO and EC treated as related but distinct resting-state conditions
- Report prediction is exploratory only

## Things I need to understand fully
- Exact target loading and merge logic
- Exact EEG preprocessing done before feature extraction
- Exact band definitions and naming conventions
- How age/gender are included in the dataset
- How regional features are appended
- How benchmark outputs are structured
