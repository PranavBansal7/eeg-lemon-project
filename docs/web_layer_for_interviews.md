# Web Layer For Interviews

## 1. What the backend does in this repo

The backend is a small FastAPI service in `backend/app/`.

Its job is not to retrain models. Its job is to:

- load the saved model from `models/rf_model.pkl`
- load the saved feature schema from `processed/feature_columns.json`
- accept user input from the UI
- convert that input into the model's expected feature vector
- return prediction results as JSON

Important repo-specific detail:

- The backend works from a small set of anchor EEG inputs, not full raw EEG files.
- The main anchor channels are `cz`, `f3`, `f4`, `fz`, and `o1`.
- It supports `eo` and `ec` inputs across the standard bandpower bands.

The main routes are:

- `GET /health`
- `POST /predict-manual`
- `POST /predict-from-csv`
- `GET /predict-demo`
- `POST /predict-from-pdf`
  This is currently experimental and returns a placeholder response, not a real prediction.

## 2. What the frontend does in this repo

The frontend is a small Next.js UI in `frontend/app/`.

Its job is to:

- collect user input
- let the user upload a CSV or PDF
- let the user enter manual EO/EC values
- call the backend API with `fetch`
- show predictions or error messages

It does not contain the ML logic. The real prediction work happens in the backend.

Important repo-specific detail:

- The main UI file is `frontend/app/HomeClient.tsx`.
- It stores form values, files, loading state, errors, and results with `useState`.
- It calls the backend through `NEXT_PUBLIC_API_BASE_URL` or `/api`.

## 3. End-to-end flow

### Manual prediction

1. The user enters `age`, `gender`, and EO/EC band values for the anchor channels.
2. The frontend builds a JSON payload and sends it to `POST /predict-manual`.
3. FastAPI validates the payload against `ManualPredictionRequest`.
4. `PredictionService` converts the anchor values into the full model feature vector.
5. The saved model returns four target predictions.
6. The backend sends back JSON like:
   `{"predictions": {"working_memory": ..., "attention": ..., ...}}`
7. The frontend renders the prediction card.

### CSV prediction

1. The user uploads a CSV file from the UI.
2. The frontend sends it as `multipart/form-data` to `POST /predict-from-csv`.
3. The backend reads the CSV, checks that rows exist, and parses `age`, `gender`, and EEG feature columns.
4. Each row is converted into anchor inputs, then into a model feature vector.
5. The backend predicts all rows in a batch.
6. The response returns row-indexed predictions.
7. The frontend shows a batch results view.

Repo-specific CSV detail:

- The backend accepts column names like `cz_alpha_eo` or similar normalized forms.
- If required EEG columns are missing, the API returns a clear error.

### Demo prediction

1. The user clicks the demo button.
2. The frontend sends `GET /predict-demo`.
3. The backend creates built-in demo anchor values.
4. It runs the same prediction pipeline used by manual input.
5. The backend returns one prediction object.
6. The frontend shows it as a demo result.

This is useful in interviews because it proves the API/UI path without requiring manual entry or a CSV file.

## 4. Minimal concepts to know

### API endpoint

An API endpoint is just a URL path the frontend can call.

Examples in this repo:

- `/predict-manual`
- `/predict-from-csv`
- `/predict-demo`

### Request/response

A request is what the frontend sends to the backend.

A response is what the backend sends back.

In this repo:

- manual prediction sends a JSON request
- CSV prediction sends a file upload request
- the backend responds with JSON

### JSON payload

A JSON payload is structured text sent in a request body.

In this repo, manual prediction sends a JSON object with:

- `age`
- `gender`
- `eo`
- `ec`

Inside `eo` and `ec`, the payload contains channel names and band values.

### File upload

A file upload means the frontend sends a file using `FormData`.

In this repo:

- CSV uploads go to `/predict-from-csv`
- PDF uploads go to `/predict-from-pdf`

### Schema validation

Schema validation means checking that input matches the expected shape and types.

In this repo, FastAPI and Pydantic validate manual prediction payloads using the models in `backend/app/schemas.py`.

That is why invalid ages or missing fields return structured errors instead of crashing silently.

### CORS

CORS is the browser rule that controls whether one web origin can call another.

In this repo, the FastAPI app enables CORS with very open settings:

- `allow_origins=["*"]`

Interview-safe explanation:

"I enabled CORS so the frontend could call the backend during local development."

### useState

`useState` is how the frontend stores UI state.

In this repo it stores things like:

- age
- gender
- manual input values
- selected files
- loading state
- errors
- results

### fetch

`fetch` is the browser API used to call the backend.

In this repo it is used to:

- `POST` manual JSON
- `POST` CSV/PDF files
- `GET` the demo prediction

## 5. Questions an interviewer might ask and simple answers

### "Why did you build a backend at all?"

"So the trained model and feature-building logic stay on the server side, and the UI just sends inputs and displays predictions."

### "What does the backend really do?"

"It loads the saved model artifacts once at startup, validates incoming data, converts anchor EEG inputs into the full feature vector the model expects, and returns predictions."

### "What does the frontend really do?"

"It is a thin client. It gathers input, uploads files, calls the API with fetch, and renders results and errors."

### "How does manual prediction differ from CSV prediction?"

"Manual prediction sends one JSON payload for one subject. CSV prediction uploads a file and returns predictions for multiple rows."

### "Why is there a demo route?"

"It gives a fast, controlled example that exercises the same backend prediction pipeline without needing user data."

### "How do you validate bad input?"

"Manual JSON is validated with Pydantic schemas, and CSV rows are checked by parser code before prediction runs."

### "What is the hardest repo-specific backend idea to explain?"

"The backend does not receive a full raw EEG recording. It receives a small set of anchor EEG values and expands them into the saved model's full feature schema."

## 6. What not to claim about this web layer

Do not claim this is:

- a production-ready web platform
- a secure deployed system with auth, users, roles, or payments
- a real PDF-to-prediction pipeline
- a raw EEG upload and processing platform
- a full-stack product with database-backed workflows
- a clinical interface

Safer phrasing:

"This is a lightweight demo layer around a saved EEG benchmark model. Its purpose is to show how the trained artifacts can be exposed through an API and simple UI."
