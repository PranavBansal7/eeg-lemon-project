# Interview Notes

## What Problem Does the Project Solve?

This project asks a practical ML question:

"Can simple, reproducible resting-state EEG features predict a small set of cognitive targets at all, and which feature design works best?"

It is not trying to claim a clinical product. It is trying to build an honest, reproducible benchmark around EEG feature engineering and baseline regression.

## Why EEG Bandpower?

Bandpower is a good starting point because it is:

- simple to compute
- common in EEG work
- easy to explain
- easy to store in a tabular ML pipeline

It turns long EEG recordings into a manageable set of numbers per channel and frequency band.

That makes it a good fit for baseline benchmarking, even if it is not the richest possible feature family.

## Why EO and EC Separately?

EO means eyes open and EC means eyes closed.

They are both resting-state conditions, but they are not identical. Keeping them separate lets the model learn differences between the two states instead of losing that information by averaging too early.

That is one of the clearest domain-aware design choices in the project.

## Why These Models?

The default suite is intentionally small:

- `dummy` gives a trivial baseline, so I know whether the model is doing anything useful at all.
- `ridge` gives a simple linear baseline.
- `random_forest` gives a stronger nonlinear tabular baseline that can handle interactions without a lot of feature scaling complexity.

The repo also supports `elasticnet` and `hist_gb`, but those are optional extensions rather than the main story.

## Why Subject-Level Fixed Splits?

The unit of prediction is the subject, not an EEG segment.

So train/test splits should also happen at the subject level.

Fixed split manifests help because:

1. the same subject never leaks into both train and test in one fold
2. reruns stay comparable over time
3. feature or model changes can be compared on the same partitions

That is a small but important reproducibility choice.

## What Were the Results?

The short answer is: modest, target-dependent, and honest.

In the saved benchmark run with `random_forest + eo_ec_concat_plus_diff_plus_regions`:

- `attention` reached mean CV R2 of about `0.07`
- `executive_function` reached mean CV R2 of about `0.13`
- `working_memory` was near `0`
- `intelligence` stayed negative

So I would say:

"There is some signal for attention and executive function, but the current feature family is weak for working memory and intelligence."

I would not oversell these results as strong prediction.

## What Were the Limitations?

- The current feature set is mostly bandpower plus basic metadata.
- The results are modest.
- There is no external held-out dataset in the current benchmark.
- The dataset is filtered to subjects who have EO, EC, metadata, and all targets.
- The report-style prediction path is exploratory only.

The clean limitation statement is:

"This is a reproducible benchmark with modest predictive signal, not a finished high-performance or clinical system."

## What Would You Improve Next?

The next improvements I would talk about are:

1. add richer EEG features beyond bandpower, such as connectivity, complexity, or aperiodic features
2. do more interpretation and error analysis to understand where the small amount of signal is coming from
3. add a stronger held-out evaluation setup beyond the current fixed CV benchmark
4. compare a few more strong tabular methods only if the feature family improves first

The key point is that the current bottleneck looks more like feature informativeness than lack of a fancy model.

## 30-Second Version

"I built a reproducible EEG regression benchmark using resting-state eyes-open and eyes-closed recordings. The core work was converting EEG into bandpower features, keeping EO and EC separate, benchmarking a few interpretable feature variants on fixed subject-level cross-validation splits, and saving all run metadata and outputs for reproducibility. The results were modest but honest: attention and executive function showed some signal, while working memory and intelligence were weak with the current feature set."

## 1-Minute Version

"This project is an EEG feature-engineering and benchmarking repo rather than a product. I take resting-state EEGLAB recordings, compute channel-wise Welch PSD bandpower features for both eyes-open and eyes-closed conditions, add age and gender, and build a subject-level regression dataset for four cognitive targets. Then I benchmark a small set of models, mainly dummy, ridge, and random forest, across fixed subject-level splits and a few explainable feature variants like EO+EC concatenation, regional summaries, and EO-minus-EC differences. The strongest saved setup was random forest with EO/EC concat plus differences plus regional features. Even then, the performance was modest: executive function and attention had small positive signal, while working memory and intelligence did not. So the strongest part of the project is the reproducible ML pipeline and honest benchmarking, not headline model accuracy."

## 2-Minute Version

"I framed this project as a reproducible EEG benchmark around a simple question: how much predictive signal can we recover from resting-state eyes-open and eyes-closed EEG using explainable tabular features? The data comes in as paired EO and EC EEGLAB files plus metadata and cognitive target tables. For each subject, the training script loads both recordings, keeps EEG channels, applies average reference, estimates the spectrum with Welch PSD, and converts that into bandpower features like delta, theta, alpha, and beta power per channel. That gives me a subject-level table of EEG features, and I append age and gender as context features.

From there, I benchmark a small public set of feature variants. The clean baseline is EO and EC concatenated as separate feature spaces. Then I add regional summaries, and then EO-minus-EC difference features. I compare those variants using fixed subject-level cross-validation splits, which are saved to disk so the exact folds can be reused across runs. The default benchmark suite is also intentionally small: dummy as a floor, ridge as a linear baseline, and random forest as a nonlinear tabular baseline. The benchmark writes metadata, fold-level results, and summary CSVs so the run is reproducible and easy to explain.

The result is an honest but modest benchmark. The best saved setup was random forest with EO/EC concat plus differences plus regional summaries. Executive function and attention had small positive R2, while working memory was near the noise floor and intelligence stayed negative. So in interviews I position this project as strong in feature engineering, experiment design, reproducibility, and clear tradeoff thinking. I do not position it as a high-accuracy or clinical prediction system."

## Fast Answers to Follow-Up Questions

### "What is the most important technical choice you made?"

"Keeping EO and EC separate, then explicitly benchmarking whether differences and regional summaries helped."

### "What is the most important engineering choice you made?"

"Saving fixed subject-level split manifests and run metadata so every comparison is repeatable."

### "What is the most honest way to describe the outcome?"

"Useful as a reproducible EEG benchmark with some signal in a couple targets, but not strong enough to claim robust prediction across all targets."

### "Why is this a good interview project even if the accuracy is modest?"

"Because it shows domain-aware feature engineering, careful baseline design, reproducibility, honest evaluation, and the ability to say what did not work."
