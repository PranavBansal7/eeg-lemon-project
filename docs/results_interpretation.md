# Results Interpretation

This note is for explaining the saved benchmark runs in a way that is positive, accurate, and interview-usable.

## Short Version

The project shows a benchmark-first result:

- `attention` and `executive_function` show modest but encouraging signal
- `working_memory` and `intelligence` remain more limited in the current setup
- `random_forest` is a better fit than `ridge` for the current tabular feature space
- more complex ratio and asymmetry features did not become a stronger main story than the simpler default feature family

That makes the repo valuable as a reproducible EEG benchmark and feature-comparison project, and best interpreted as a benchmark-first comparison rather than a general cognitive predictor.

## Main Saved Result

The clearest saved benchmark story comes from:

- run: `results/benchmark_v1/benchmark_v1_20260402T004104Z`
- model: `random_forest`
- feature variant: `eo_ec_concat_plus_diff_plus_regions`
- dataset size: 200 subjects
- evaluation: fixed 5-fold subject-level CV

Mean CV R2 from that run:

| Target | Mean CV R2 | Interpretation |
| --- | ---: | --- |
| `attention` | `0.072` | modest but encouraging signal |
| `executive_function` | `0.133` | strongest target so far, still modest |
| `working_memory` | `-0.020` | limited signal in the current setup |
| `intelligence` | `-0.103` | currently limited signal with these features |

## What These Numbers Mean

The cleanest way to say it is:

"The project shows modest but non-zero signal on selected targets, especially attention and executive function. The current feature family is still more limited for working memory and intelligence. That makes the benchmark useful for comparing feature and model choices, even though the result is not a broad predictive claim."

## Model Comparison

In `results/benchmark_v1/benchmark_v1_20260401T232221Z`, `ridge` and `random_forest` were compared across the public feature variants.

What stands out:

- `ridge` stays negative across all four targets and all public variants
- `random_forest` consistently reaches the strongest positive R2 values for `attention` and `executive_function`

Interview-safe takeaway:

"For this feature space, the nonlinear tabular baseline was a better fit than the simple linear baseline."

## Why Random Forest Likely Looks Stronger Here

A good plain-English explanation is:

- the inputs are tabular, not images or raw sequences
- EO, EC, region summaries, and metadata can interact in nonlinear ways
- ridge can only learn one linear weighting pattern
- random forest can use thresholds and interactions more naturally

Important honesty note:

This is not proof that random forest is always the best model.

It is an explanation for why it is a better fit in this current benchmark.

## Feature Variant Comparison

The most interview-friendly comparison is among:

- `eo_ec_concat`
- `eo_ec_concat_plus_regions`
- `eo_ec_concat_plus_diff_plus_regions`

Why this comparison matters:

- it keeps the feature story easy to explain
- it tests whether regional summaries help
- it tests whether explicit EO-minus-EC contrast helps
- it keeps the main benchmark focused on interpretable changes

The gains are not huge, but the point of the benchmark is that the comparison is explicit and reproducible.

## Why the More Complex Features Are Not the Main Story

In `results/benchmark_v1/benchmark_v1_20260402T002127Z`, more complex variants were tested:

- `eo_ec_concat_plus_diff_plus_regions_plus_ratios`
- `eo_ec_concat_plus_diff_plus_regions_plus_ratios_plus_asymmetry`

Those runs are useful, but they did not become the strongest external story.

Examples from the saved summaries:

- `attention` drops from `0.072` in the simpler best variant to negative R2 in the ratio and ratio-plus-asymmetry variants
- `executive_function` stays positive, but below the simpler benchmark setup

That is still a useful outcome because it shows disciplined ablation:

"More handcrafted feature complexity did not automatically improve the best targets."

## How to Interpret the Results

- There is modest but non-zero signal for some targets.
- The benchmark is useful for comparing feature variants and model choices.
- The project is an encouraging direction for a student benchmark project.
- The current numbers are best interpreted as benchmark-first results.
- The value is not just the best score. The value is also the reproducible design, the fair comparisons, and the clarity about what did and did not help.

## If An Interviewer Asks About Strengths And Limits

Strongest part:

"The strongest part is the reproducible benchmark design: clear feature variants, fixed folds, fair comparisons, and saved outputs that make the results easy to inspect."

More limited part:

"The current feature family is still simple, so some targets remain more limited. That is why the project is strongest as a benchmark-first comparison rather than a broad prediction claim."

## Good Ways to Describe the Outcome

Good phrasing:

- "The results are modest but encouraging on attention and executive function."
- "Random forest performed better than ridge on the current feature space."
- "The project was useful for comparing feature variants and model choices in a reproducible way."
- "The strongest value comes from disciplined benchmarking rather than headline accuracy."

Phrasing to avoid:

- "We solved cognitive prediction from EEG."
- "The model is highly accurate."
- "The project proves strong cognitive prediction."

## Bottom Line

This is a strong benchmark project because it combines:

- domain-aware EEG feature engineering
- reproducible subject-level split handling
- interpretable feature-variant comparisons
- honest reporting of modest but encouraging signal on selected targets

That is a solid interview outcome even though the predictive performance is still modest overall.
