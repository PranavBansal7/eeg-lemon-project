# Metrics And Model Reading Guide

This doc explains how to read the saved benchmark outputs in this repository.

## Why This Repo Uses Multiple Metrics

No single metric tells the whole story.

This repo reports:

- `R2`
- `RMSE`
- `MAE`
- `Pearson correlation`

Together, they give a better picture of model quality.

## R2

R2 answers:

"How much better is this model than predicting the mean?"

In this repo:

- positive R2 means the model beats a mean baseline on held-out data
- near-zero R2 means the model is only recovering a very small amount of useful signal
- negative R2 means the model does worse than a mean baseline on held-out data

Why R2 matters here:

- it is the easiest way to compare models within the same target
- it makes the benchmark story easy to summarize

## RMSE

RMSE means root mean squared error.

It answers:

"How large are the prediction errors, in the original units of the target?"

Why it matters here:

- it stays in raw target space
- it makes error size easier to interpret than scaled-space metrics

## MAE

MAE means mean absolute error.

It answers:

"On average, how far off are the predictions?"

Why it matters here:

- it is often easier to interpret than RMSE
- it is less influenced by a few very large errors

## Pearson Correlation

Pearson correlation measures whether predictions move in the same direction as the true values.

Plain-English version:

"When the true target goes up, does the prediction usually go up too?"

Why it matters here:

- it can show directional alignment even when R2 is still modest
- it is a useful supporting metric, not the main headline metric

## How to Interpret R2 in This Project

Use this rule of thumb:

- clearly positive R2 on held-out folds
  there is some recoverable signal
- around zero
  signal is minimal in the current setup
- negative
  the current model and feature space are not beating a mean baseline for that target

Repo-specific examples from the best saved run:

- `attention`: about `0.072`
- `executive_function`: about `0.133`
- `working_memory`: about `-0.020`
- `intelligence`: about `-0.103`

That is why the right summary is:

"Modest but encouraging signal on selected targets, with more limited signal on others."

## How to Compare Models Fairly

Compare models fairly by holding everything else fixed:

1. same target
2. same feature variant
3. same split manifest
4. same evaluation space

Good comparison:

"For `executive_function` on `eo_ec_concat_plus_diff_plus_regions`, random forest outperformed ridge on the same saved folds."

Less useful comparison:

"This model had a lower RMSE on one target than another target."

That is less useful because different targets are on different scales.

## How to Read `summary_results.csv`

Each row in `summary_results.csv` represents:

- one experiment name
- one run ID
- one model
- one feature variant
- one target

Important columns:

- `model_name`
- `feature_variant`
- `target`
- `r2_mean`
- `r2_std`
- `rmse_mean`
- `mae_mean`
- `pearson_r_mean`
- `n_folds`

The main questions to ask when reading it are:

1. which model is strongest for this target?
2. which feature variant is strongest for this target?
3. are the gains large or only small?
4. is the result stable enough across folds to be worth discussing?

## How to Describe "Modest But Encouraging Signal" Correctly

In this repo, "modest but encouraging signal" means:

- R2 is positive on held-out folds
- the result is still far from a strong predictive system
- the outcome is useful for comparing modeling choices

A good sentence is:

"Attention and executive function showed modest but encouraging signal, which makes the benchmark useful for comparing feature variants and model choices even though the overall predictive strength is still limited."

## A Good Reading Strategy

Start in this order:

1. check `model_name`, `feature_variant`, and `target`
2. read `r2_mean`
3. check `r2_std`
4. use `rmse_mean` and `mae_mean` as supporting context
5. use `pearson_r_mean` as a secondary directional check

## Bottom Line

For this repository, the fairest summary is:

"The benchmark is most valuable as a reproducible way to compare feature variants and baseline models. Positive R2 on attention and executive function is encouraging, while the more limited targets show that the current feature family still has clear limitations."
