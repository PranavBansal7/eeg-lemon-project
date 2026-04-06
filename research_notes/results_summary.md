
### `research_notes/results_summary.md`
```markdown
# Results Summary

## Current best setup
- Best model: `random_forest`
- Best feature variant: `eo_ec_concat_plus_diff_plus_regions`
- Evaluation protocol:
  - fixed subject-level CV splits
  - fold-safe target scaling
  - metrics: R², RMSE, MAE, Pearson r

## Stable findings
- Random forest strongly outperforms ridge on the current EEG tabular feature space.
- HistGradientBoosting did not beat random forest.
- Best feature variant so far is `eo_ec_concat_plus_diff_plus_regions`.
- Attention and executive function show modest predictive signal.
- Working memory is near the noise floor.
- Intelligence remains poorly predicted with current bandpower-based features.
- Adding ratio-based features and frontal alpha asymmetry did not improve the strongest targets.
- Report-based prediction should be treated as exploratory only.

## Target-wise view

### Attention
- Currently one of the strongest targets.
- Best performance is modest but positive.
- EO/EC combination plus regional summaries is useful.
- Ratios/asymmetry reduced performance.

### Executive function
- Currently the strongest target overall.
- Responds best to the current RF + concat/diff/regions setup.
- Ratios/asymmetry did not improve it.

### Working memory
- Near the noise floor.
- No convincing predictive signal yet.
- Added feature complexity has not helped.

### Intelligence
- Still poorly predicted.
- Some variants slightly reduce negativity, but performance remains unconvincing.
- Current setup does not support strong claims here.

## Negative findings
- Ridge performed poorly across all targets.
- Ratio-based features reduced performance for attention and executive function.
- Frontal alpha asymmetry did not improve the benchmark.
- HistGradientBoosting did not beat RandomForest.

## Current interpretation
The current bottleneck appears to be feature informativeness / signal strength rather than simply lack of a stronger generic tabular model. More feature complexity did not automatically help, so future work should emphasize interpretability, controlled ablation, and better understanding of where the signal actually comes from.

## Current limitations
- Bandpower-focused feature family
- Modest sample size
- No connectivity / aperiodic / complexity features yet
- No XGBoost / LightGBM comparison yet
- Report-based prediction is exploratory only

## Next priorities
1. Understand the codebase top to bottom
2. Build a clear project map in plain language
3. Add interpretation-oriented analysis
4. Consider feature importance and error analysis
5. Revisit stronger boosting later only if justified