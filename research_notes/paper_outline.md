# Paper Outline

## Working title
NeuroScope: EEG-Based Cognitive Prediction from Resting-State Signals

## Core question
Can resting-state EO/EC EEG features predict cognitive targets in a reproducible ML benchmark setup?

## Motivation
Resting-state EEG is relatively easy to collect and may contain weak but measurable associations with cognitive traits. A careful benchmark can help determine which targets are realistically predictable and which feature families are actually useful.

## Dataset
- LEMON EEG dataset
- Resting-state EEG
- EO and EC conditions
- Subject-level dataset construction

## Targets
- working_memory
- attention
- executive_function
- intelligence

## Feature families tested so far
- EO-only bandpower
- EC-only bandpower
- EO+EC concatenation
- EO-EC difference
- EO/EC log-ratio
- regional mean bandpower summaries
- ratio-based band features
- frontal alpha asymmetry

## Models tested so far
- Dummy baseline
- Ridge
- ElasticNet
- Random Forest
- HistGradientBoosting

## Evaluation protocol
- Fixed subject-level CV splits
- Fold-safe target scaling
- Metrics:
  - R²
  - RMSE
  - MAE
  - Pearson r

## Current findings
- Random forest is the strongest baseline so far.
- Best feature variant is `eo_ec_concat_plus_diff_plus_regions`.
- Attention and executive function show modest predictive signal.
- Working memory remains near the noise floor.
- Intelligence remains difficult to predict.
- Ratio-based features and frontal alpha asymmetry did not improve the strongest targets.
- HistGradientBoosting did not outperform RandomForest.

## Main interpretation
The project currently suggests that modest signal exists for some cognitive targets, but improvements depend more on disciplined EO/EC feature construction and controlled benchmarking than on simply increasing feature complexity.

## Limitations
- Bandpower-based features only
- Limited model family comparison so far
- No connectivity / spectral slope / complexity measures yet
- No stronger external boosting libraries yet
- Prediction/report use should remain exploratory only

## Future work
- Feature importance analysis
- Error analysis
- Better project understanding and method explanation
- Possibly XGBoost / LightGBM later
- More principled feature families if justified by current findings

## Claim discipline
- Avoid overstating predictive power
- Emphasize reproducibility and ablation
- Frame report-style prediction as exploratory only