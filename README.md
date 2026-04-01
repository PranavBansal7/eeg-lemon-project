# NeuroScope: EEG-Based Cognitive Prediction from Resting-State Signals

### A machine learning pipeline and web application for predicting cognitive task outcomes from resting-state EEG using EO/EC bandpower features, metadata, and report-adapted inference.

## 1. Project Title
NeuroScope: EEG-Based Cognitive Prediction from Resting-State Signals

## 2. Project Overview
NeuroScope is an end-to-end machine learning project that predicts multiple cognitive and behavioral outcomes from resting-state EEG recordings. The pipeline ingests EEGLAB .set files, extracts EO and EC bandpower features, integrates metadata signals such as age and gender, and trains a multi-target regression model for simultaneous prediction of:

- working_memory
- attention
- executive_function
- intelligence

Beyond model training, the project includes a production-style web interface and API for manual, batch, demo, and report-adapted personal inference workflows.

## 3. Key Features
- End-to-end EEG ML workflow from raw/resting-state signals to deployable inference.
- Support for paired Eyes Open and Eyes Closed recordings.
- Automated feature extraction from EEGLAB .set files using channel-wise bandpower.
- Metadata fusion with age and gender.
- Multi-target regression using RandomForestRegressor and MultiOutputRegressor.
- Cross-validated evaluation with per-target metrics.
- Artifact persistence for reproducibility: model, feature schema, processed tables, and target scaler.
- Report-adapted personal prediction script for anchor-channel inference.
- FastAPI backend with health, manual, CSV, PDF (experimental), and demo endpoints.
- Next.js and React frontend with TypeScript-based prediction workflows.

## 4. Tech Stack Used
- Python
- MNE
- NumPy
- pandas
- scikit-learn
- joblib
- FastAPI
- Next.js / React
- TypeScript
- GitHub Codespaces
- Git / GitHub

## 5. Dataset and Data Sources
This project is structured around resting-state EEG and phenotype data from the LEMON-style directory layout present in this repository:

- EEG recordings: data/eeg/EEG_Preprocessed_BIDS_ID
	- EEGLAB files such as subject_EO.set and subject_EC.set
- Metadata and phenotype tables: data/phenotype/Behavioural_Data_MPILMBB_LEMON
	- Demographics: age and gender
	- Cognitive target tables for working memory, attention, executive function, and intelligence

## 6. Machine Learning Pipeline
The core ML pipeline is implemented in src/train_lemon_multitarget.py and emphasizes robust data assembly and multi-output modeling.

Pipeline stages:

1. Load metadata and target tables.
2. Match subjects with both EO and EC EEG recordings.
3. Read .set files and preprocess EEG channels.
4. Compute PSD-based bandpower features by channel and frequency band.
5. Concatenate EO and EC features and append metadata (age, gender).
6. Build target matrix for four cognitive outcomes.
7. Optionally standardize targets to z-scores.
8. Train and evaluate a multi-output random forest model.
9. Persist model and feature artifacts for deployment.

## 7. EO and EC Feature Handling
EO and EC are handled as distinct feature spaces during training.

- Each EEG channel is represented with state-specific features, for example:
	- channel_alpha_eo
	- channel_alpha_ec
- During model training, EO and EC features are both included in the final input vector.
- In report-adapted inference, known anchor channels are used to estimate full-channel feature vectors:
	- frontal channels estimated from f3, f4, fz
	- central channels estimated from cz
	- posterior/occipital channels estimated from o1
	- remaining channels estimated from global averages

This design preserves EO/EC signal differences while enabling practical inference from sparse report inputs.

## 8. Model Training and Evaluation
Modeling strategy:

- Base model: RandomForestRegressor
- Multi-target wrapper: MultiOutputRegressor
- Validation: KFold cross-validation
- Metrics per target: R2 and RMSE

Training objective:

- Learn shared and target-specific relationships between resting-state bandpower and behavioral outcomes.
- Generate one model that predicts all four cognitive targets in a single inference pass.

Saved training artifacts:

- models/rf_model.pkl
- processed/features.csv
- processed/feature_columns.json
- processed/target_scaler.json

## 9. Report-Adapted Personal Prediction
The script src/predict_my_report.py supports a report-adapted personal prediction workflow.

What it does:

- Loads the trained model and feature schema.
- Accepts EO and EC anchor-band values from EEG report-style summaries.
- Expands sparse anchors into a model-compatible full feature vector.
- Adds metadata (age and gender).
- Produces exploratory predictions for all four targets.

This is designed for technical experimentation and personalized model probing, not clinical use.

## 10. Backend and Frontend Overview
Backend:

- Implemented with FastAPI in backend/app.
- Loads model and feature schema at startup.
- Exposes endpoints:
	- GET /health
	- POST /predict-manual
	- POST /predict-from-csv
	- POST /predict-from-pdf (experimental placeholder)
	- GET /predict-demo

Frontend:

- Implemented with Next.js, React, and TypeScript in frontend/app.
- Provides four user workflows:
	- Upload CSV (recommended)
	- Upload EEG report PDF (experimental)
	- Manual EO/EC entry (advanced)
	- Demo prediction

## 11. Folder Structure

		eeg-lemon-project/
		├── backend/
		│   ├── app/
		│   │   ├── main.py
		│   │   ├── schemas.py
		│   │   └── services/
		│   └── requirements.txt
		├── data/
		│   ├── eeg/
		│   └── phenotype/
		├── frontend/
		│   ├── app/
		│   │   ├── HomeClient.tsx
		│   │   ├── page.tsx
		│   │   └── globals.css
		│   ├── package.json
		│   └── next.config.ts
		├── models/
		│   └── rf_model.pkl
		├── my_data/
		│   └── assessment_*.csv
		├── processed/
		│   ├── features.csv
		│   ├── feature_columns.json
		│   └── target_scaler.json
		├── src/
		│   ├── train_lemon_multitarget.py
		│   └── predict_my_report.py
		├── requirements.txt
		└── README.md

## 12. How to Run
### A. Train the ML pipeline

1. Install Python dependencies:

		pip install -r requirements.txt

2. Run training:

		python src/train_lemon_multitarget.py

3. Run report-adapted inference:

		python src/predict_my_report.py

### B. Run the backend

1. Install backend dependencies:

		python -m pip install -r backend/requirements.txt

2. Start FastAPI:

		cd backend
		python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

### C. Run the frontend

1. Install frontend dependencies:

		cd frontend
		npm install

2. Start Next.js:

		npm run dev -- --hostname 0.0.0.0 --port 3000

3. Open:

- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs

## 13. Outputs Generated
Primary outputs generated by training and inference workflows:

- models/rf_model.pkl
- processed/features.csv
- processed/feature_columns.json
- processed/target_scaler.json

## 14. Disclaimer
This project is intended for machine learning research, technical demonstration, and educational use only. Predictions are exploratory and not clinically validated. They must not be used for medical diagnosis, treatment planning, or healthcare decision-making.

## 15. Resume-Friendly Summary
NeuroScope demonstrates applied machine learning and deployment skills across the full EEG analytics lifecycle:

- Built a reproducible multi-target regression pipeline for resting-state EEG using EO and EC bandpower features.
- Engineered metadata-aware feature sets and standardized target handling for robust model training.
- Implemented and evaluated RandomForestRegressor with MultiOutputRegressor for simultaneous cognitive outcome prediction.
- Delivered production-style inference interfaces through a FastAPI backend and Next.js/React TypeScript frontend.
