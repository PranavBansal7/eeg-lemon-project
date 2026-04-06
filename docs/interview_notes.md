# Interview Notes

Use this file to practice the project story out loud.

The strongest framing is:

"I built a reproducible EEG feature-engineering and regression benchmark on resting-state eyes-open and eyes-closed data."

## The Core Story

This project answers a practical ML question:

"If I turn resting-state EEG into simple, explainable tabular features, which feature design and baseline model choices recover the most useful signal?"

That gives you a strong interview angle:

- it is a real signal-processing-to-ML pipeline
- it has a clear ablation and benchmarking story
- it saves artifacts and split definitions for reproducibility
- it stays honest about modest predictive performance

## What I Built

The shortest accurate answer is:

"I built a subject-level EEG benchmarking pipeline that reads paired EO and EC recordings, converts them into bandpower features, compares a small set of interpretable feature variants and regression baselines on fixed subject-level splits, and saves all run outputs for reproducibility."

## Why EEG Bandpower?

Bandpower is a strong starting point here because it is:

- simple to compute
- common enough to be recognizable in EEG work
- easy to explain in plain language
- easy to store in a tabular ML pipeline

It turns a long EEG recording into a manageable set of channel-by-band features.

## Why EO and EC Separately?

EO means eyes open and EC means eyes closed.

They are both resting-state conditions, but they are not identical. Keeping them separate gives the model a chance to use the state contrast directly instead of losing that information by averaging too early.

That is one of the clearest domain-aware design choices in the project.

## Why These Models?

The default suite is intentionally small:

- `dummy`
  A floor. It tells me whether I am beating a trivial baseline.
- `ridge`
  A simple linear baseline for the tabular feature space.
- `random_forest`
  A nonlinear tabular baseline that can capture interactions without a lot of extra modeling machinery.

The repo also supports `elasticnet` and `hist_gb`, but those are optional extensions rather than the main story.

## Why Subject-Level Fixed Splits?

The prediction unit is the subject, not an EEG segment.

So train/test splits should also happen at the subject level.

Fixed split manifests matter because:

1. the same subject never leaks into both train and test within a fold
2. reruns stay comparable over time
3. feature or model changes can be judged on the same partitions

## What Were the Main Results?

The short answer is:

"Modest, target-dependent, and useful as a benchmark."

In the saved benchmark run with `random_forest + eo_ec_concat_plus_diff_plus_regions`:

- `attention` reached mean CV R2 of about `0.07`
- `executive_function` reached mean CV R2 of about `0.13`
- `working_memory` was about `-0.02`
- `intelligence` was about `-0.10`

Good interpretation:

"There is modest but encouraging signal for attention and executive function. Working memory and intelligence still look more limited with the current feature set. That makes the repo strongest as a reproducible benchmark and feature-comparison project."

## What Is Technically Interesting Here?

The most interesting parts are:

- turning paired EO and EC EEG recordings into a clean subject-level ML table
- keeping EO and EC separate instead of collapsing them too early
- comparing feature variants that stay interpretable
- using fixed split manifests and train-fold-only target scaling
- saving enough metadata that runs can be audited later

## What Were the Limitations?

- The feature family is still mostly bandpower plus basic metadata.
- Results are modest and target-dependent.
- There is no external held-out test set in the current benchmark.
- The dataset is filtered to subjects who have EO, EC, metadata, and all targets.
- The web layer and sparse-input paths are useful extensions, but they are not the central evaluated path.

Clean limitation sentence:

"This is a reproducible benchmark with modest predictive signal, not a finished high-performance or clinical system."

## What Would I Improve Next?

Best next-step answer:

1. add richer EEG features beyond bandpower
2. do more feature and error analysis on the strongest targets
3. add a stronger held-out evaluation setup
4. only expand model complexity once the feature family improves

That answer keeps the emphasis on feature informativeness and disciplined experimentation.

## 30-Second Version

"I built a reproducible EEG regression benchmark using resting-state eyes-open and eyes-closed recordings. The core work was converting EEG into bandpower features, keeping EO and EC separate, benchmarking a few interpretable feature variants on fixed subject-level splits, and saving all run metadata and outputs for reproducibility. The results were modest but encouraging on attention and executive function, and more limited on working memory and intelligence, so I position it as a strong benchmark-first ML project."

## 1-Minute Version

"This project is an EEG feature-engineering and benchmarking repo rather than a product. I take resting-state EEGLAB recordings, compute channel-wise Welch PSD bandpower features for both eyes-open and eyes-closed conditions, add age and gender, and build a subject-level regression dataset for four cognitive targets. Then I benchmark a small set of models, mainly dummy, ridge, and random forest, across fixed subject-level splits and a few explainable feature variants like EO-plus-EC concatenation, regional summaries, and EO-minus-EC differences. The strongest saved setup was random forest with EO/EC plus differences plus regional features. Even then, the performance was modest, so the strongest part of the project is the reproducible ML pipeline, the feature comparison story, and the honest benchmarking."

## 2-Minute Version

"I framed this project as a reproducible EEG benchmark around a simple question: how much predictive signal can we recover from resting-state eyes-open and eyes-closed EEG using explainable tabular features? The data comes in as paired EO and EC EEGLAB files plus metadata and cognitive target tables. For each subject, the training script loads both recordings, keeps EEG channels, applies average reference, estimates the spectrum with Welch PSD, and converts that into bandpower features like delta, theta, alpha, and beta power per channel. That gives me a subject-level table of EEG features, and I append age and gender as context features.

From there, I benchmark a small public set of feature variants. The clean baseline is EO and EC concatenated as separate feature spaces. Then I add regional summaries, and then EO-minus-EC difference features. I compare those variants using fixed subject-level cross-validation splits, which are saved to disk so the exact folds can be reused across runs. The default benchmark suite is also intentionally small: dummy as a floor, ridge as a linear baseline, and random forest as a nonlinear tabular baseline. The benchmark writes metadata, fold-level results, and summary CSVs so the run is reproducible and easy to explain.

The result is a benchmark-first outcome. The best saved setup was random forest with EO/EC plus differences plus regional summaries. Executive function and attention showed modest but encouraging positive signal, while working memory and intelligence remained more limited in the current setup. So in interviews I position this project as strong in feature engineering, reproducibility, and clear model-comparison thinking rather than as a high-accuracy prediction product."

## Fast Answers to Likely Follow-Ups

### "What is the most important technical choice you made?"

"Keeping EO and EC separate, then explicitly benchmarking whether regional summaries and EO-minus-EC differences helped."

### "What is the most important engineering choice you made?"

"Saving fixed subject-level split manifests and benchmark metadata so every comparison is repeatable."

### "What is the most honest way to describe the outcome?"

"Useful as a reproducible EEG benchmark with modest but encouraging signal on a couple targets, and best interpreted as a benchmark-first result."

### "Why is this a good interview project even if the scores are modest?"

"Because it shows domain-aware feature engineering, careful baselines, reproducibility, honest evaluation, and the ability to say what helped and what did not."

### "What should I say about the web layer?"

"It is a lightweight demo layer on top of saved artifacts. It shows engineering breadth, but it is not the main validated contribution."

## Preferred Phrases

Good phrases to use:

- "reproducible EEG benchmark"
- "benchmark-first result"
- "modest but encouraging signal"
- "interpretable feature variants"
- "subject-level fixed splits"
- "useful for comparing feature and model choices"

Phrases to avoid:

- "highly accurate cognitive prediction"
- "clinical model"
- "production EEG platform"
- "solved cognition from EEG"
