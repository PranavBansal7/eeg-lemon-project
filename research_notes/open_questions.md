# Open Questions

## Code understanding
- Exactly where are targets loaded and merged into the subject-level table?
- How are EO and EC file pairs discovered and validated?
- What preprocessing is done to EEG before bandpower extraction?
- How exactly are bandpower features computed?
- Are age and gender included in every benchmark feature variant?
- Where is target scaling done inside the benchmark flow?
- Where are metrics computed and aggregated?

## Feature engineering
- Why did ratio-based features hurt attention and executive function?
- Is RF already capturing most of the useful nonlinear structure from the current base features?
- Would feature pruning help more than adding more derived ratios?
- Are there cleaner region definitions worth testing later?

## Modeling
- Why does random forest outperform ridge so strongly here?
- Why did HistGradientBoosting underperform RF?
- Would XGBoost / LightGBM behave differently enough to matter?
- Should future work stay target-specific rather than forcing all four targets equally?

## Research framing
- What is the fairest way to describe “modest predictive signal”?
- How should I explain negative R² cleanly in a project discussion?
- What claims are safe for attention/executive function?
- How should I frame intelligence and working memory results honestly?

## Next analysis ideas
- Feature importance
- Error analysis
- Target-wise scatter/residual inspection
- Better interpretation of EO vs EC contributions
- More disciplined ablation summaries