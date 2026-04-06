# Interview Notes

Use this file to practice how you would explain the project in ML and SWE interviews.

The goal is not to memorize polished lines.

The goal is to understand the project well enough that you can explain it clearly, defend the design choices, and answer follow-up questions without sounding scripted.

## Best One-Sentence Framing

"I built a reproducible EEG feature-engineering and regression benchmark on resting-state eyes-open and eyes-closed data."

That sentence is short, honest, and strong.

## What Problem Does This Repo Solve?

Good answer:

"The repo turns resting-state EEG into a clean tabular ML benchmark so I can compare feature designs and baseline models in a reproducible way. The main question is how much useful signal can be recovered when EO and EC are kept separate and evaluated on fixed subject-level splits."

## What Did I Actually Build?

Good answer:

"I built a subject-level pipeline that reads paired EO and EC EEG recordings, converts them into Welch PSD bandpower features, adds age and gender, benchmarks a small set of interpretable feature variants and regression baselines, and saves all fold-level and run-level outputs for reproducibility."

## The Main Story To Defend

The strongest things to emphasize are:

- the EEG-to-tabular feature pipeline is clear and explainable
- EO and EC are kept separate on purpose
- the benchmark design is reproducible
- the default feature/model suite is intentionally small and readable
- the results are modest but still useful for comparison

## Likely ML/SWE Interview Questions

### What problem does this project solve?

Short answer:

"It gives a reproducible way to compare EEG feature variants and baseline regressors on resting-state EO/EC data."

### What happens from raw data to final metrics?

Short answer:

"The code matches usable subjects, preprocesses EO and EC files, computes bandpower features, builds subject-level rows, creates feature variants, runs fixed subject-level folds, predicts held-out targets, and saves raw-space metrics plus run metadata."

### Why use EEG bandpower features?

Short answer:

"Bandpower is a simple and explainable starting point. It compresses a long EEG recording into channel-by-band summary features that work well in a tabular benchmark."

### Why keep EO and EC separate?

Short answer:

"Because they are different resting-state conditions. Keeping them separate preserves state information and lets the model use EO-versus-EC contrast directly."

### Why these models?

Short answer:

"I wanted a small, explainable suite: dummy as the floor, ridge as the linear baseline, and random forest as the stronger nonlinear tabular baseline."

### Why is random forest stronger than ridge here?

Short answer:

"The current feature space is tabular and likely contains nonlinear interactions across EO, EC, regional summaries, and metadata. Random forest can use those interactions more naturally than a linear model like ridge."

Important honesty note:

This is an explanation of fit to this feature space, not a claim that random forest is always the best model.

### Why save split manifests?

Short answer:

"So every later comparison uses the exact same subject-level train/test partitions. That makes result differences easier to trust."

### How do you avoid leakage?

Short answer:

"The benchmark splits at the subject level, not at the segment level, and target scaling is fit on the train fold only before predictions are inverse-transformed back to raw space."

### What are the strongest and weakest parts of the project?

Strongest part:

"The strongest part is the benchmark design: clear features, fixed splits, fair comparisons, and saved outputs that make the results easy to audit."

More limited part:

"The current feature family is still fairly simple, so the predictive performance is modest and target-dependent."

### What is the benchmark path versus the optional demo path?

Short answer:

"The benchmark path is the core ML contribution. The demo path is the optional FastAPI and Next.js layer that serves saved artifacts and shows API/UI integration."

## Honest Result Story

The clean result summary is:

- `attention` and `executive_function` show modest but encouraging signal
- `working_memory` and `intelligence` remain more limited in the current setup
- `random_forest` performs better than `ridge` on the current feature family
- more complex ratio and asymmetry variants did not become the strongest main story

Good way to say it:

"The strongest value lies in reproducible comparison and disciplined evaluation, with modest but encouraging signal on selected targets."

## Good Technical Choices To Highlight

If an interviewer asks what decisions you are proud of, good answers are:

- keeping EO and EC separate instead of averaging too early
- setting up a small feature-variant ladder instead of one opaque feature table
- saving fixed subject-level split manifests
- doing train-fold-only target scaling
- saving metadata and CSV outputs for every run

## 30-Second Version

"I built a reproducible EEG regression benchmark using resting-state eyes-open and eyes-closed recordings. The core work was turning EEG into bandpower features, keeping EO and EC separate, benchmarking a small set of feature variants and baseline models on fixed subject-level splits, and saving all run outputs for reproducibility. The results were modest but encouraging on attention and executive function, so I frame it as a strong benchmark-first ML project."

## 1-Minute Version

"This project is a benchmark-first EEG ML repo rather than a product. I start from paired EO and EC EEGLAB recordings, compute channel-wise Welch PSD bandpower features, add age and gender, and build a subject-level regression dataset for four cognitive targets. Then I compare a small set of readable feature variants and baseline models, mainly dummy, ridge, and random forest, on fixed subject-level folds that are saved to disk. The strongest saved setup was random forest with EO/EC plus regional summaries plus EO-minus-EC differences. The predictive performance is still modest, so the strongest story is the reproducible pipeline, the clear feature comparison setup, and the honest evaluation."

## 2-Minute Version

"I framed this project around a simple question: how much predictive signal can I recover from resting-state eyes-open and eyes-closed EEG using explainable tabular features? The data comes in as paired EO and EC EEGLAB files plus metadata and target tables. For each subject, the training script loads both recordings, keeps EEG channels, applies average reference, estimates the spectrum with Welch PSD, and converts that into bandpower features like alpha and beta power per channel. That gives me a subject-level table of EEG features, and I append age and gender as simple context features.

From there, I benchmark a small public set of feature variants. The clean baseline is EO and EC concatenated as separate feature spaces. Then I add regional summaries, and then EO-minus-EC difference features. I compare those variants on fixed subject-level cross-validation folds that are saved to disk so every rerun uses the same partitions. The default model suite is also intentionally small: dummy as a floor, ridge as a linear baseline, and random forest as a nonlinear tabular baseline. The benchmark writes metadata, fold-level results, and summary CSVs so the run is reproducible and easy to explain.

The outcome is benchmark-first. The best saved setup was random forest with EO/EC plus differences plus regional summaries. Executive function and attention showed modest but encouraging positive signal, while working memory and intelligence remained more limited in the current setup. So in interviews I position the project as strong in feature engineering, reproducibility, and clear model-comparison thinking rather than as a high-accuracy prediction product."

## If They Ask You To Walk Through The Code

A good order is:

1. `run_resume_benchmark.py`
2. `src/benchmark_v1.py`
3. `src/train_lemon_multitarget.py`
4. `src/feature_variants.py`
5. `src/split_manifest.py`

Why this order:

- it starts with the simplest entrypoint
- then shows the benchmark runner
- then shows where the base dataset comes from
- then explains how feature variants are built
- then explains how reproducible folds are frozen

## Preferred Phrases

Good phrases:

- "reproducible EEG benchmark"
- "clear and explainable feature-engineering pipeline"
- "benchmark-first ML project"
- "modest but encouraging signal on selected targets"
- "subject-level fixed splits"
- "strongest value lies in reproducible comparison and disciplined evaluation"
- "useful engineering breadth through the optional demo layer"

Phrases to avoid:

- "highly accurate cognitive prediction"
- "clinical system"
- "production EEG platform"
- "solved cognition from EEG"
- "the web app is the main contribution"
