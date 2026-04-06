# ML Concepts For This Project

This doc explains only the ML ideas that matter for this repository.

## What a Baseline Model Is

A baseline model is a simple reference point.

Its job is to answer:

"Is the more interesting model actually doing anything better than something simple?"

This repo uses a small baseline suite on purpose so the comparisons stay easy to explain.

## What `DummyRegressor` Means Here

`DummyRegressor` is the floor.

In this repo it predicts the mean target value from the training data.

Why it matters:

- if a real model cannot beat the dummy baseline, that target probably has very limited usable signal in the current feature space
- it gives you a sanity check before talking about more complex models

## What Ridge Means Here

Ridge is a linear regression model with L2 regularization.

Plain-English version:

"It is a simple linear model that tries not to overreact to many correlated features."

In this repo, Ridge is wrapped in a `Pipeline` with `StandardScaler`.

Why it is included:

- it is a clean linear baseline
- it tells you whether simple weighted sums of the features are enough

## What Random Forest Means Here

Random Forest is a tree-based ensemble model.

Plain-English version:

"It builds many decision trees and averages them, which lets it model nonlinear relationships and feature interactions."

In this repo, the main configuration is:

- 500 trees
- `max_features="sqrt"`
- fixed `random_state`

## Why Random Forest Makes Sense Here

Random Forest makes sense here because:

- the input is tabular
- EO/EC features, region means, and metadata may interact nonlinearly
- it does not need the same kind of feature scaling that linear models do
- it is still easy to explain as a strong tabular baseline

The saved results also support it: it performs better than Ridge on the current feature space.

## What Train/Test Leakage Means

Leakage means information from the test set sneaks into training.

That makes results look better than they really are.

Examples of leakage to avoid:

- using the same subject in both train and test
- computing scaling statistics from the full dataset instead of the training fold
- tuning decisions based on test information

This repo tries to avoid leakage by using subject-level split manifests and train-fold-only target scaling in the benchmark.

## Why Train-Fold-Only Scaling Matters

In `src/benchmark_v1.py`, target scaling is done on the train fold only.

That matters because:

- the model is trained in scaled space
- the test fold should not influence the scaling parameters
- predictions are inverse-transformed back to raw target space before metrics are computed

Plain-English version:

"The test fold is treated like future unseen data, so its statistics should not be used when preparing the training problem."

## Why Subject-Level Splits Matter

The prediction unit in this repo is the subject.

So the split unit should also be the subject.

If the same subject appeared in both train and test, the benchmark would be misleading.

That is why `src/split_manifest.py` works with subject IDs rather than EEG segments.

## What Feature Ablation Means

Feature ablation means comparing feature sets on purpose instead of only reporting one final feature table.

In this repo, that means testing:

- EO and EC together as separate subspaces
- EO and EC plus region summaries
- EO and EC plus region summaries plus EO-minus-EC differences

Why it matters:

- it helps explain what each feature idea adds
- it turns the project into a benchmark instead of a single-model guess

## What Reproducibility Means in This Repo

Reproducibility here means:

- fixed subject-level split manifests
- explicit target-column locking
- saved benchmark metadata
- timestamped output folders
- saved fold-level and summary-level CSVs
- the same default command for the main benchmark path

Plain-English version:

"If I rerun this project later, I can tell whether a result changed because of a real code change or just because the setup drifted."

## The Main ML Story in One Paragraph

"This repo is not trying to show the fanciest model. It is trying to show careful feature engineering, fair baseline comparisons, leakage-aware evaluation, and reproducible benchmarking on a real EEG tabular problem."
