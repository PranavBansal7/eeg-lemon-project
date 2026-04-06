# Web Layer For Interviews

This doc covers only the backend and frontend ideas that matter for this repository.

The preferred framing is:

"The web layer is a lightweight demo around saved EEG benchmark artifacts. It is secondary to the benchmark, but it is still useful because it shows API design, schema validation, and UI wiring around a trained model."

## 1. What the Backend Does in This Repo

The backend lives in `backend/app/` and is built with FastAPI.

Its job is not to retrain anything. Its job is to:

- load the saved model from `models/rf_model.pkl`
- load the saved feature schema from `processed/feature_columns.json`
- accept user input from the UI
- convert that input into the feature vector the model expects
- return prediction results as JSON

Important repo-specific detail:

- the backend works from a small set of anchor EEG inputs, not full raw EEG files
- the main anchor channels are `cz`, `f3`, `f4`, `fz`, and `o1`
- inputs are grouped by condition: `eo` and `ec`
- each condition contains values for the standard bandpower bands

Main files:

- `backend/app/main.py`
  defines the routes
- `backend/app/services/predictor.py`
  loads artifacts and runs predictions
- `backend/app/services/feature_builder.py`
  expands anchor inputs into the full saved feature schema

## 2. What the Frontend Does in This Repo

The frontend lives in `frontend/app/` and is built with Next.js.

Its job is to:

- collect user input
- upload CSV or PDF files
- let the user enter manual EO/EC anchor values
- call the backend with `fetch`
- display predictions, info messages, and validation errors

Important repo-specific detail:

- the main UI file is `frontend/app/HomeClient.tsx`
- the render is now split into small local components inside `HomeClient.tsx` so
  the workflow cards, optional manual-entry section, and result panel are easier to study
- it stores form state, file state, loading state, errors, and results with `useState`
- it calls the backend through `NEXT_PUBLIC_API_BASE_URL` or `/api`

The frontend does not contain the ML logic. The real prediction work happens in the backend.

## 3. End-to-End Flow

### Manual prediction

1. The user enters `age`, `gender`, and EO/EC band values for the anchor channels.
2. The frontend builds a JSON payload and sends it to `POST /predict-manual`.
3. FastAPI validates the payload with the `ManualPredictionRequest` schema.
4. `PredictionService` normalizes the anchors and asks `FeatureBuilder` for a full feature vector.
5. The saved model returns four target predictions.
6. The backend sends JSON back to the frontend.
7. The frontend renders the single-subject prediction card.

### CSV prediction

1. The user uploads a CSV file from the UI.
2. The frontend sends it as `multipart/form-data` to `POST /predict-from-csv`.
3. The backend parses each row into `age`, `gender`, and anchor EEG inputs.
4. Each row is expanded into a full feature vector.
5. The backend predicts all rows in a batch.
6. The response returns row-indexed predictions.
7. The frontend renders a results table.

### Demo prediction

1. The user clicks the demo button.
2. The frontend calls `GET /predict-demo`.
3. The backend creates built-in example anchor values.
4. It runs the same prediction path used by manual input.
5. The backend returns one prediction object.
6. The frontend renders the example result.

### PDF prediction

1. The user uploads a PDF.
2. The frontend sends it to `POST /predict-from-pdf`.
3. The backend checks that it is a non-empty PDF.
4. The current implementation returns a structured acknowledgement response.

This route is best described as a future extension rather than a full prediction workflow.

## 4. Minimal Concepts to Know

### API endpoint

An API endpoint is just a URL path the frontend can call.

Examples in this repo:

- `/predict-manual`
- `/predict-from-csv`
- `/predict-demo`

### Request and response

A request is what the frontend sends.

A response is what the backend sends back.

In this repo:

- manual prediction sends JSON
- CSV and PDF flows send file uploads
- the backend responds with JSON

### JSON payload

A JSON payload is structured text sent in the request body.

Here, manual prediction sends:

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

Schema validation means checking that input has the expected shape and type.

In this repo, FastAPI and Pydantic validate manual prediction payloads before prediction runs. That is why invalid ages or missing fields return structured errors instead of crashing silently.

### CORS

CORS is the browser rule that decides whether one web origin can call another.

In this repo, FastAPI enables CORS broadly so the frontend can call the backend during local development.

### useState

`useState` is how the frontend stores UI state.

In this repo it stores:

- age
- gender
- manual EEG inputs
- selected files
- loading state
- errors
- results

### fetch

`fetch` is the browser API used to call the backend.

In this repo it is used to:

- `POST` manual JSON
- `POST` CSV and PDF files
- `GET` the demo prediction

## 5. Local Setup Note

The main local setup idea is:

- let the browser call `/api`
- let Next.js rewrite that request to the backend

That keeps the frontend code simple and avoids hard-coding a backend URL in the browser by default.

### `BACKEND_INTERNAL_URL`

This variable is used in `frontend/next.config.ts`.

Purpose:

- tell Next.js where the backend actually lives when rewriting `/api/:path*`

Current default:

- `http://127.0.0.1:8000`

Good mental model:

"This is the internal target for the Next.js proxy rewrite."

### `NEXT_PUBLIC_API_BASE_URL`

This variable is used in `frontend/app/HomeClient.tsx`.

Purpose:

- override the browser-side fetch base if you want the client to call a direct API URL instead of `/api`

Current behavior:

- if unset, the frontend uses `/api`
- if set, the frontend calls that value directly

Good mental model:

"This is the browser-visible base URL for fetch."

### Recommended Local Flow in This Repo

Recommended local setup:

1. start the backend on port `8000`
2. leave `NEXT_PUBLIC_API_BASE_URL` unset
3. optionally leave `BACKEND_INTERNAL_URL` unset too, since it already defaults to `http://127.0.0.1:8000`
4. run the frontend on port `3000`

Then the flow is:

- browser calls `http://127.0.0.1:3000/api/...`
- Next.js rewrites that request to `http://127.0.0.1:8000/...`
- the backend responds with JSON

That is the cleanest local story to explain in interviews.

## 6. Questions an Interviewer Might Ask

### "Why build a backend at all?"

"So the saved model artifacts and feature-building logic stay on the server side, and the UI just sends inputs and displays results."

### "What does the backend actually do?"

"It loads the saved artifacts once at startup, validates incoming data, expands anchor EEG inputs into the full feature vector the model expects, and returns predictions."

### "What does the frontend actually do?"

"It is a thin client. It gathers input, uploads files, calls the API with fetch, and renders results and errors."

### "How does manual prediction differ from CSV prediction?"

"Manual prediction sends one JSON payload for one subject. CSV prediction uploads a file and returns predictions for multiple rows."

### "Why is there a demo route?"

"It gives a fast, controlled example that exercises the same backend path without needing user data."

### "What is the hardest backend idea to explain?"

"The backend does not receive a full raw EEG recording. It receives a small set of anchor EEG values and expands them into the saved model's feature schema."

## 7. Preferred Framing

Good ways to describe this web layer:

- a lightweight demo around saved EEG benchmark artifacts
- a way to show artifact serving, schema validation, and UI integration
- strongest on the manual, CSV, and demo prediction paths
- inclusive of a PDF route as a future extension
- interview-secondary compared with the benchmarked ML workflow

One safe summary sentence:

"This web layer is a lightweight interface around a saved EEG benchmark model. Its value is showing how trained artifacts can be exposed through an API and simple UI without changing the core benchmark story."
