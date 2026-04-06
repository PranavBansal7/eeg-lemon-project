# Results Interpretation

This note is meant to help explain the saved benchmark runs in a way that is honest, constructive, and interview-usable.

## Short Version

The project does show useful signal, but only for some targets.

- `attention` and `executive_function` show modest but non-zero signal.
- `working_memory` and `intelligence` remain weak with the current feature family.
- `random_forest` is clearly more appropriate than `ridge` for the current tabular EEG setup.
- More complex ratio and asymmetry features did not become the strongest main story.

That makes the repo valuable as a reproducible EEG benchmark and feature-comparison project, even though it is not strong enough for broad cognitive prediction claims.

## Main Saved Results

In `results/benchmark_v1/benchmark_v1_20260402T004104Z`, the best saved story is:

- model: `random_forest`
- feature variant: `eo_ec_concat_plus_diff_plus_regions`
- dataset size: 200 subjects
- evaluation: fixed 5-fold subject-level CV

Mean CV R2 from that run:

- `attention`: `0.072`
- `executive_function`: `0.133`
- `working_memory`: `-0.020`
- `intelligence`: `-0.103`

Plain-English interpretation:

- `executive_function` is the strongest target so far.
- `attention` is also encouraging, though still modest.
- `working_memory` is close to the noise floor.
- `intelligence` is not reliably predicted with the current features.

## Model Comparison

In `results/benchmark_v1/benchmark_v1_20260401T232221Z`, `ridge` and `random_forest` were compared across the public feature variants.

What stands out:

- `ridge` stays strongly negative across all four targets and all public variants.
- `random_forest` consistently reaches small positive R2 for `attention` and `executive_function`.

That supports a clean interview point:

"For this feature space, a simple linear model was too weak, while random forest gave a better nonlinear tabular baseline."

## Feature Variant Comparison

The most interview-friendly comparison is among:

- `eo_ec_concat`
- `eo_ec_concat_plus_regions`
- `eo_ec_concat_plus_diff_plus_regions`

These variants are close in performance, but they support a useful design story:

- keeping EO and EC separate is reasonable
- adding regional summaries is interpretable
- adding EO-minus-EC differences gives a clean way to represent state contrast

Even when the gains are small, the benchmark is useful because it makes those comparisons explicit and reproducible.

## Why the More Complex Features Are Not the Main Story

In `results/benchmark_v1/benchmark_v1_20260402T002127Z`, more complex variants were tested:

- `eo_ec_concat_plus_diff_plus_regions_plus_ratios`
- `eo_ec_concat_plus_diff_plus_regions_plus_ratios_plus_asymmetry`

Those runs did not produce a better main story.

Examples from the saved summaries:

- `attention` drops from `0.072` in the simpler best variant to negative R2 in the ratio and ratio-plus-asymmetry variants.
- `executive_function` stays positive, but below the simpler benchmark setup.

So the honest takeaway is:

"More handcrafted EEG feature complexity did not automatically improve the strongest targets."

That is actually a useful result for interviews because it shows disciplined ablation rather than feature inflation.

## How To Interpret These Results

- There is modest but non-zero signal for some targets.
- The benchmark is useful for comparing models and feature variants.
- The project is in an encouraging direction for a student benchmark project.
- The current numbers are not strong enough for broad cognitive prediction claims.
- The value of the project is not just the final score. It is also the reproducible experiment design, the careful baseline comparisons, and the ability to explain what did and did not help.

## Safe Interview Phrasing

Good phrasing:

- "The results are modest but encouraging on attention and executive function."
- "The benchmark shows some signal, but not enough to claim robust prediction across all targets."
- "Random forest performed better than ridge on the current feature space."
- "The project was useful for comparing feature variants and model choices in a reproducible way."

Phrasing to avoid:

- "We solved cognitive prediction from EEG."
- "The model is highly accurate."
- "The project proves strong cognitive prediction from resting-state EEG."

## Bottom Line

This is a strong benchmark project because it combines:

- domain-aware EEG feature engineering
- reproducible train/test split handling
- interpretable feature-variant comparisons
- honest reporting of limited but non-zero signal

That is a solid outcome for interviews, even though the predictive performance is still modest.
