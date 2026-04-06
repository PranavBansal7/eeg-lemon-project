# Feature Variant Cheatsheet

This doc explains why feature variants exist and which ones matter most in interviews.

## Why Feature Variants Exist

The point of feature variants is not feature inflation.

The point is to test a small set of design choices explicitly:

- should EO and EC be kept separate?
- do regional summaries help?
- does explicit EO-minus-EC contrast help?
- do more handcrafted ratios or asymmetry features help enough to justify the added complexity?

That is why this repo is strongest as a feature-comparison benchmark.

## The 3 Public / Default Variants

These are the interview-friendly default variants in `src/feature_variants.py`.

### `eo_ec_concat`

What it does:

- keeps EO features and EC features as separate columns

Why it matters:

- it is the clean baseline
- it preserves both resting-state conditions without mixing them too early

### `eo_ec_concat_plus_regions`

What it does:

- starts from `eo_ec_concat`
- adds region-level means such as frontal, central, parietal, occipital, and temporal summaries

Why it matters:

- it adds a simpler anatomical view without removing channel-level detail

### `eo_ec_concat_plus_diff_plus_regions`

What it does:

- starts from `eo_ec_concat`
- adds EO-minus-EC difference features
- adds region-level summaries

Why it matters:

- it gives the model raw EO values, raw EC values, and explicit state contrast
- it is the clearest best-current benchmark story

## Optional Exploratory Variants

These still exist and still work, but they are demoted out of the main story.

Exploratory variants:

- `eo_only`
- `ec_only`
- `eo_ec_diff`
- `eo_ec_logratio`
- `eo_ec_concat_plus_diff`
- `eo_ec_concat_plus_logratio`
- `eo_ec_concat_plus_diff_plus_regions_plus_ratios`
- `eo_ec_concat_plus_diff_plus_regions_plus_ratios_plus_asymmetry`

These are useful for:

- ablations
- curiosity-driven checks
- showing that you tested more than one idea

They are not the best first explanation for this project.

## Best Variants to Discuss in Interviews

If you only discuss three, discuss these:

1. `eo_ec_concat`
2. `eo_ec_concat_plus_regions`
3. `eo_ec_concat_plus_diff_plus_regions`

That gives you a clean progression:

- baseline
- add interpretable summaries
- add explicit EO-versus-EC contrast

## Comparison Table

| Variant | What It Adds | Why It Exists | Best Interview Use |
| --- | --- | --- | --- |
| `eo_ec_concat` | EO and EC side by side | clean baseline | starting point |
| `eo_ec_concat_plus_regions` | regional means | test whether simpler summaries help | explain interpretability |
| `eo_ec_concat_plus_diff_plus_regions` | EO-EC differences + regions | test explicit state contrast | strongest main story |
| `eo_ec_diff` | EO minus EC only | isolate pure contrast | optional ablation |
| `eo_ec_logratio` | log EO/EC contrast | test normalized contrast | exploratory only |
| ratios / asymmetry variants | more handcrafted EEG heuristics | test whether extra complexity helps | mention briefly as exploratory |

## How to Explain the Variant Strategy

Good interview answer:

"I did not just build one feature table and hope it worked. I set up a small feature-ablation ladder: first keep EO and EC separate, then add regional summaries, then add EO-minus-EC differences. That made it easy to compare feature families in a reproducible way and keep the main story interpretable."

## What the Saved Results Suggest

The simpler public/default variants are the right main story.

Why:

- they are easy to explain
- they stay interpretable
- the best saved run comes from `eo_ec_concat_plus_diff_plus_regions`
- more complex ratio and asymmetry features did not become the strongest overall result

That makes the public/default variant set a better interview story than the full exploratory list.
