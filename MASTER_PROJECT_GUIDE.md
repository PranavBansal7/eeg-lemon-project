# NeuroScope Master Guide

## NeuroScope: EEG-Based Cognitive Prediction from Resting-State Signals

A complete step-by-step study guide for understanding this project end-to-end, including machine learning, backend, frontend, and interview preparation.

---

## Who This Guide Is For

This guide is written for you if:

1. You are comfortable with Python basics but new to backend/frontend development.
2. You want to explain your project confidently in interviews.
3. You need a practical and detailed understanding of what each file does.

---

## 1) Project Goal in One Paragraph

NeuroScope predicts four cognitive outcomes from resting-state EEG:

1. working_memory
2. attention
3. executive_function
4. intelligence

It does this by extracting EO and EC bandpower features from EEGLAB `.set` files, adding metadata (`age`, `gender`), training a multi-target ML model (`RandomForestRegressor` inside `MultiOutputRegressor`), and exposing predictions through a FastAPI backend and Next.js frontend.

---

## 2) End-to-End Architecture (Big Picture)

1. Raw EEG + phenotype/cognitive CSV data are read from `data/`.
2. Training script builds features and target matrix.
3. Model is trained and saved in `models/rf_model.pkl`.
4. Feature schema is saved in `processed/feature_columns.json`.
5. Backend loads these artifacts once at startup.
6. Frontend calls backend endpoints for manual, CSV, PDF placeholder, and demo predictions.

Simple flow:

`EEG files -> feature engineering -> multi-target model -> API -> web app`

---

## 3) Repository Roadmap (What to Study First)

Recommended file reading order:

1. `src/train_lemon_multitarget.py`
2. `src/predict_my_report.py`
3. `backend/app/main.py`
4. `backend/app/services/predictor.py`
5. `backend/app/services/feature_builder.py`
6. `frontend/app/HomeClient.tsx`
7. `frontend/next.config.ts`

This order moves from core ML to deployment.

---

## 4) Dataset and Data Organization

The project assumes LEMON-style structure:

1. EEG files in `data/eeg/EEG_Preprocessed_BIDS_ID`.
2. Metadata and target CSVs in `data/phenotype/...`.
3. Each subject ideally has both:
   - EO file (`*_EO.set`)
   - EC file (`*_EC.set`)

Important data integration rule:

A subject is used only if all required pieces exist:

1. EO EEG
2. EC EEG
3. metadata (`age`, `gender`)
4. all target values

Why this matters: it prevents partial-row training noise and inconsistent supervision.

---

## 5) Machine Learning Core (Most Important Section)

This is the strongest part of your project. Learn this section deeply.

### 5.1 Training Entry Point

Main file: `src/train_lemon_multitarget.py`

Core outputs:

1. `models/rf_model.pkl`
2. `processed/features.csv`
3. `processed/feature_columns.json`
4. `processed/target_scaler.json`

### 5.2 EEG Bandpower Features

Bands used:

1. delta: 1-4 Hz
2. theta: 4-8 Hz
3. low_alpha: 8-10 Hz
4. alpha: 8-12 Hz
5. high_alpha: 10-12 Hz
6. low_beta: 12-20 Hz
7. beta: 12-30 Hz
8. high_beta: 20-30 Hz

Bandpower is computed by PSD integration (Welch PSD):

$$
Bandpower = \int_{f_{low}}^{f_{high}} PSD(f)\, df
$$

### 5.3 Why EO and EC Are Separate

Features are generated as separate names:

1. `channel_band_eo`
2. `channel_band_ec`

This keeps state information intact. EO and EC are different physiological conditions, so merging too early can lose predictive signal.

### 5.4 Metadata Fusion

Two metadata features are appended:

1. `age`
2. `gender`

This gives the model useful covariates that EEG alone may not capture.

### 5.5 Target Setup

The project is **multi-target regression** with 4 targets in one pipeline:

1. working_memory
2. attention
3. executive_function
4. intelligence

### 5.6 Model Choice

Model in code:

```python
base = RandomForestRegressor(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_features="sqrt",
)
model = MultiOutputRegressor(base)
```

Interpretation:

1. `RandomForestRegressor` learns nonlinear relationships in tabular features.
2. `MultiOutputRegressor` wraps one regressor per target under one interface.
3. You get one training/prediction pipeline for all targets.

### 5.7 Evaluation

Validation strategy:

1. KFold cross-validation (`CV_SPLITS = 5`)
2. Metrics per target:
   - R2
   - RMSE

If `STANDARDIZE_TARGETS = True`, targets are transformed to z-scores before training:

$$
z = \frac{y - \mu}{\sigma}
$$

This helps when target scales differ.

### 5.8 Core Training Loop (Conceptual)

```text
load metadata + targets
-> find EO/EC pairs
-> preprocess EEG
-> PSD + bandpower features
-> append age/gender
-> build X, y
-> cross-validate
-> fit final model on all data
-> save model + schema + scaler
```

### 5.9 Key Interview Point

Say this clearly:

"I designed a domain-aware tabular ML pipeline where EO and EC are separate feature subspaces, then trained a multi-target random forest for simultaneous cognitive prediction with per-target cross-validated metrics."

---

## 6) EEG Preprocessing in Your Code

Implemented in `load_and_preprocess_eeg(...)`:

1. Read EEGLAB with MNE.
2. Keep EEG channels only.
3. Apply average reference.

Why this is practical:

1. Simple and reproducible.
2. Works well for spectral feature extraction.
3. Easy to justify in interviews.

---

## 7) Feature Engineering Deep Dive

Function: `compute_bandpower_features(raw, state_suffix)`

What it does:

1. Extract EEG data matrix.
2. Compute PSD in 1-30 Hz.
3. Integrate PSD in each predefined band.
4. Create feature names like `f3_alpha_eo`, `o1_beta_ec`.

Why this is strong:

1. Interpretable features.
2. Noise-resistant vs raw time-series direct modeling.
3. Good baseline for medium dataset sizes.

---

## 8) Report-Adapted Personal Prediction (Important Practical Innovation)

Main file: `src/predict_my_report.py`

Use case: personal EEG report may only provide anchor channels (cz, f3, f4, fz, o1).

Your script handles this by:

1. Loading full feature schema.
2. Using known anchor values for EO/EC.
3. Estimating missing channels via region templates:
   - frontal -> average of f3/f4/fz
   - central -> cz
   - posterior -> o1
   - others -> global average
4. Building complete model input row.
5. Predicting all 4 targets.

This is excellent to discuss in interviews because it solves real-world sparse-input problems.

---

## 9) Backend Explained for Beginners

Main backend file: `backend/app/main.py`

### 9.1 What FastAPI Does Here

FastAPI provides HTTP endpoints so frontend or tools can request predictions.

### 9.2 Startup Pattern

In the app lifespan:

1. Load model and feature schema once.
2. Save service object in app state.
3. Reuse it for all requests.

Why this matters: avoids reloading model every request.

### 9.3 Endpoints

1. `GET /health` -> service health
2. `GET /` -> root info
3. `POST /predict-manual` -> direct EO/EC values
4. `POST /predict-from-csv` -> batch predictions
5. `POST /predict-from-pdf` -> experimental placeholder
6. `GET /predict-demo` -> built-in demo input

### 9.4 Prediction Service Layer

File: `backend/app/services/predictor.py`

Responsibilities:

1. Load artifacts (`rf_model.pkl`, `feature_columns.json`)
2. Normalize incoming anchors
3. Build feature vectors using `FeatureBuilder`
4. Predict single or batch
5. Map raw model output to named target keys

### 9.5 Feature Builder Layer

File: `backend/app/services/feature_builder.py`

Responsibilities:

1. Parse schema names (`*_eo`, `*_ec`, plain)
2. Fill age/gender directly
3. Fill existing anchor features directly
4. Estimate missing channels with region rules

This keeps backend behavior aligned with report-adapted inference logic.

---

## 10) Frontend Explained for Beginners

Main UI file: `frontend/app/HomeClient.tsx`

### 10.1 What the Frontend Does

It gives 4 workflows:

1. Upload CSV
2. Upload PDF (experimental)
3. Manual EO/EC entry
4. Demo prediction

### 10.2 Why `API_BASE = "/api"`

In browser-forwarded/devcontainer URLs, direct `localhost:8000` can fail from client context.

So frontend calls same-origin `/api/...`, and Next.js rewrites proxy to backend.

Rewrite in `frontend/next.config.ts`:

```ts
async rewrites() {
  const backendBaseUrl = process.env.BACKEND_INTERNAL_URL || "http://127.0.0.1:8000";
  return [
    {
      source: "/api/:path*",
      destination: `${backendBaseUrl}/:path*`,
    },
  ];
}
```

This is a practical full-stack engineering detail interviewers like.

### 10.3 State Management Basics

The component tracks:

1. Inputs (`age`, `gender`, EO/EC matrix)
2. Selected files
3. Active workflow loading state
4. Error state
5. Results state (single, batch, info)

---

## 11) Full Request Flow (Frontend to Model)

Example: user clicks Demo.

1. Frontend calls `GET /api/predict-demo`.
2. Next.js rewrite forwards to backend `GET /predict-demo`.
3. FastAPI endpoint prepares demo anchors.
4. `PredictionService` builds model feature vector.
5. Model predicts 4 targets.
6. JSON response returns to frontend.
7. Frontend displays results cards.

This is your complete MLOps-lite story.

---

## 12) Runbook (From Zero)

### 12.1 Train model

```bash
cd /workspaces/eeg-lemon-project
pip install -r requirements.txt
python src/train_lemon_multitarget.py
```

### 12.2 Backend

```bash
cd /workspaces/eeg-lemon-project
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 12.3 Frontend

```bash
cd /workspaces/eeg-lemon-project/frontend
npm install
npm run dev -- --hostname 0.0.0.0 --port 3000
```

### 12.4 Open

1. Frontend: http://localhost:3000
2. Backend docs: http://localhost:8000/docs

---

## 13) Common Errors and Fixes

### Error: Failed to fetch on frontend

Fix:

1. Use `/api` proxy in frontend.
2. Ensure Next rewrite points to backend.
3. Ensure backend is running on 8000.

### Error: `No module named uvicorn`

Fix:

```bash
python -m pip install -r backend/requirements.txt
```

### Error: 404 on backend root

Fix:

Use root route in `main.py` and check `GET /`.

---

## 14) Probable Interview Questions with Detailed Answers

## Q1) Explain your project in 60 seconds.

Answer:

NeuroScope is an end-to-end machine learning pipeline that predicts four cognitive outcomes from resting-state EEG. I extract EO and EC bandpower features from EEGLAB `.set` files, add age and gender, and train a multi-target model using `RandomForestRegressor` wrapped with `MultiOutputRegressor`. I evaluate with K-fold CV using per-target R2 and RMSE, save model artifacts, and deploy inference through FastAPI plus a Next.js frontend with CSV, manual, demo, and report-adapted workflows.

## Q2) Why did you choose bandpower features?

Answer:

Bandpower is robust, interpretable, and effective for resting-state EEG. It captures frequency-domain information linked to cognitive processes while reducing data dimensionality compared with raw time-series. This makes it a strong choice for tabular models and moderate dataset sizes.

Deeper version:

By integrating PSD over predefined bands, I create stable statistics less sensitive to phase shifts and local temporal noise. This improves generalization for classical regressors.

## Q3) Why separate EO and EC instead of combining early?

Answer:

EO and EC represent different physiological states. If we average them too early, we lose potentially predictive contrast. So the schema explicitly keeps `*_eo` and `*_ec` features.

## Q4) Why RandomForest + MultiOutputRegressor?

Answer:

Random forest handles nonlinear interactions and mixed feature scales without heavy normalization. `MultiOutputRegressor` lets me train one coordinated pipeline for all targets.

Deeper version:

It trains one estimator per target under a common interface. This is simple, strong, and easy to deploy. A known limitation is it does not explicitly model target correlations.

## Q5) How do you prevent data leakage?

Answer:

I merge by subject ID and keep only complete EO+EC+metadata+target rows. Feature extraction is independent of targets. Evaluation uses K-fold splits so test folds are unseen during each fold's training.

## Q6) What metrics do you report and why?

Answer:

I report R2 and RMSE for each target. R2 tells relative improvement over a mean baseline, RMSE gives absolute prediction error magnitude.

## Q7) Why standardize targets?

Answer:

Targets can have different scales. Z-score standardization keeps them comparable and often improves training stability for multi-target settings.

## Q8) What is report-adapted inference and why is it useful?

Answer:

In real reports we may only have anchor channels. Report-adapted inference expands these sparse values into a full schema-compatible input using region-based templates, enabling practical inference without full channel export.

## Q9) How does your backend work internally?

Answer:

FastAPI loads artifacts once at startup via a prediction service. Each endpoint parses input, builds full feature vectors through feature builder logic, runs model prediction, and returns structured JSON.

## Q10) How does frontend connect to backend reliably?

Answer:

Frontend calls same-origin `/api` routes. Next.js rewrites proxy these requests to backend `http://127.0.0.1:8000`. This avoids common browser localhost/CORS/preview issues.

## Q11) What are your biggest project limitations?

Answer:

1. No explicit uncertainty estimation.
2. PDF pipeline is placeholder.
3. Model family is baseline classical ML; no deep sequence model comparison yet.
4. Predictions are exploratory, not clinical.

## Q12) What would you improve first?

Answer:

1. Add stronger baselines and hyperparameter search.
2. Add confidence intervals/uncertainty estimates.
3. Implement robust PDF parser for automatic value extraction.
4. Add experiment tracking and strict reproducibility metadata.

## Q13) If interviewer asks "Is this deployable?"

Answer:

Yes as an internal research tool. It already has a service layer and UI workflows. For production-grade deployment, I would add authentication, observability, stricter validation, model versioning, and uncertainty reporting.

## Q14) What did you personally learn from this project?

Answer:

I learned how to convert a research-style EEG pipeline into a usable product: data integration, feature engineering, multi-target modeling, API design, UI workflows, and debugging real environment issues.

## Q15) Why is this project good for ML internships?

Answer:

It demonstrates complete ownership of the ML lifecycle:

1. domain-specific data processing
2. thoughtful feature engineering
3. model training and evaluation
4. reproducible artifacts
5. backend serving
6. frontend integration

---

## 15) Easy-Language + Deep-Technical Answer Pairs

### Pair A: "What is your model?"

Easy:

I use a random forest model that predicts 4 outputs at once.

Deep:

I use `MultiOutputRegressor(RandomForestRegressor(...))`, which trains one forest per target but keeps a unified training and prediction interface for multi-target regression.

### Pair B: "How do you transform EEG to ML features?"

Easy:

I convert each EEG channel into frequency-band strengths (bandpower).

Deep:

I estimate PSD with Welch in 1-30 Hz and integrate PSD over predefined band ranges to produce channel-band-state features.

### Pair C: "How do you handle missing channels in report inference?"

Easy:

I estimate missing channels from nearby known channels.

Deep:

I build EO/EC state maps, compute frontal/central/posterior/global templates from anchors, and fill missing channel-band values based on channel region rules.

---

## 16) 30-Second, 2-Minute, and 5-Minute Interview Scripts

### 30-second script

I built NeuroScope, an EEG ML pipeline that predicts four cognitive outcomes from resting-state EO/EC bandpower plus metadata. I engineered the full data-to-model workflow, evaluated with cross-validation, and deployed inference through FastAPI and a Next.js interface.

### 2-minute script

The pipeline starts by matching subjects with complete EO, EC, metadata, and target records. EEG `.set` files are loaded with MNE, preprocessed, and converted into channel-wise bandpower features across eight bands for both EO and EC states. I append age and gender and train a multi-target random forest model using `MultiOutputRegressor`. Evaluation is done with K-fold CV and per-target R2/RMSE. I then persist model and schema artifacts and expose them through FastAPI endpoints for manual input, CSV batch, and demo prediction. The frontend calls these endpoints through a stable `/api` proxy rewrite and displays formatted results for all targets.

### 5-minute script

Expand the 2-minute script with:

1. why EO/EC separation matters biologically,
2. why tabular spectral features were chosen over raw deep models,
3. report-adapted sparse-input reconstruction logic,
4. practical deployment/debug lessons.

---

## 17) Final Quick Revision Sheet (Last-Minute Interview Prep)

1. Problem type: multi-target regression.
2. Inputs: EO+EC bandpower + age + gender.
3. Targets: working_memory, attention, executive_function, intelligence.
4. Model: RandomForestRegressor + MultiOutputRegressor.
5. Metrics: R2 and RMSE per target.
6. Important artifact files:
   - models/rf_model.pkl
   - processed/feature_columns.json
   - processed/target_scaler.json
7. Backend framework: FastAPI.
8. Frontend framework: Next.js + React + TypeScript.
9. Key practical innovation: report-adapted inference from anchor channels.
10. Limitation statement: exploratory, not clinical.

---

## Closing Advice

For interviews, do not just describe tools. Explain decisions:

1. Why this feature engineering?
2. Why this model?
3. What trade-offs did you accept?
4. How did you make it usable end-to-end?

If you can explain these clearly, this project becomes a very strong ML internship and research profile.
