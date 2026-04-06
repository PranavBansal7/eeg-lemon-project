# Experiment Log

## 2026-04-02 — stable benchmark checkpoint

### Goal
Establish the strongest current baseline model and feature variant for resting-state EEG cognitive prediction.

### Setup
- Dataset: LEMON EEG
- Signal conditions:
  - EO
  - EC
- Targets:
  - attention
  - executive_function
  - intelligence
  - working_memory
- Evaluation:
  - fixed subject-level CV splits
  - fold-safe target scaling
  - metrics: R², RMSE, MAE, Pearson r

### Models tested so far
- dummy
- ridge
- elasticnet
- random_forest
- hist_gb

### Feature variants tested so far
- eo_only
- ec_only
- eo_ec_concat
- eo_ec_diff
- eo_ec_logratio
- eo_ec_concat_plus_diff
- eo_ec_concat_plus_logratio
- eo_ec_concat_plus_regions
- eo_ec_concat_plus_diff_plus_regions
- eo_ec_concat_plus_diff_plus_regions_plus_ratios
- eo_ec_concat_plus_diff_plus_regions_plus_ratios_plus_asymmetry

### Main findings
- `random_forest` is the strongest model so far.
- `eo_ec_concat_plus_diff_plus_regions` is the strongest feature variant so far.
- Attention and executive function show modest predictive signal.
- Working memory is near the noise floor.
- Intelligence remains poorly predicted with current bandpower-based features.
- Ratio-based features and frontal alpha asymmetry did not improve the strongest targets.
- `hist_gb` did not beat `random_forest`.

### Best-known current configuration
- Model: `random_forest`
- Feature variant: `eo_ec_concat_plus_diff_plus_regions`

### Interpretation
The current evidence suggests that careful EO/EC combination and regional summarization are more useful than adding more handcrafted ratio/asymmetry features. The main bottleneck appears to be feature informativeness / signal strength rather than simply lack of a more complex generic tabular model.

### Decision
- Keep `random_forest` as the main benchmark baseline.
- Keep `ridge` as a weak linear sanity-check baseline.
- Keep `hist_gb` logged as tested, but not preferred.
- Keep `eo_ec_concat_plus_diff_plus_regions` as the current best feature set.

### Next step
Understand the project top to bottom before adding more complexity.

---

## Template for future entries

## YYYY-MM-DD — short experiment title

### Goal
What question was I trying to answer?

### Setup
- Model(s):
- Feature variant(s):
- Targets:
- Split setup:
- Any special notes:

### Command run
```bash
python3 benchmark_v1.py ...

