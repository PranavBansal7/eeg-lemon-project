"use client";

import { ChangeEvent, FormEvent, useState } from "react";

// By default the browser talks to `/api`, which Next.js rewrites to the backend.
// `NEXT_PUBLIC_API_BASE_URL` is still available when we want the client to call
// a direct backend URL instead.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";

const CONDITIONS = ["eo", "ec"] as const;
const ELECTRODES = ["cz", "f3", "f4", "fz", "o1"] as const;
const BANDS = [
  "delta",
  "theta",
  "low_alpha",
  "high_alpha",
  "alpha",
  "low_beta",
  "beta",
  "high_beta",
] as const;

const TARGET_KEYS = [
  "working_memory",
  "attention",
  "executive_function",
  "intelligence",
] as const;

type Condition = (typeof CONDITIONS)[number];
type Electrode = (typeof ELECTRODES)[number];
type Band = (typeof BANDS)[number];
type TargetKey = (typeof TARGET_KEYS)[number];

type PredictionMetrics = Record<TargetKey, number>;

type CsvPredictionItem = {
  row_index: number;
  predictions: PredictionMetrics;
};

type PdfResponse = {
  message: string;
  experimental: boolean;
  filename: string;
  size_bytes: number;
  notes: string;
};

type ManualInputMatrix = Record<Condition, Record<Electrode, Record<Band, string>>>;
type ManualNumericMatrix = Record<Condition, Record<Electrode, Record<Band, number>>>;

type WorkflowKind = "csv" | "pdf" | "manual" | "demo";

type ResultState =
  | {
      mode: "single";
      title: string;
      prediction: PredictionMetrics;
    }
  | {
      mode: "batch";
      title: string;
      rows: CsvPredictionItem[];
    }
  | {
      mode: "info";
      title: string;
      message: string;
      notes: string;
    };

type FormSubmitHandler = (event: FormEvent<HTMLFormElement>) => Promise<void>;

type HeroCardProps = {
  apiBase: string;
};

type WorkflowGridProps = {
  activeWorkflow: WorkflowKind | null;
  onCsvSubmit: FormSubmitHandler;
  onPdfSubmit: FormSubmitHandler;
  onDemoClick: () => Promise<void>;
  onCsvFileSelect: (file: File | null) => void;
  onPdfFileSelect: (file: File | null) => void;
};

type OptionalManualEntryDemoProps = {
  age: string;
  gender: string;
  manualInputs: ManualInputMatrix;
  activeWorkflow: WorkflowKind | null;
  onAgeChange: (value: string) => void;
  onGenderChange: (value: string) => void;
  onManualInputChange: (
    condition: Condition,
    electrode: Electrode,
    band: Band,
    value: string
  ) => void;
  onSubmit: FormSubmitHandler;
};

type ResultsPanelProps = {
  activeWorkflow: WorkflowKind | null;
  error: string | null;
  result: ResultState | null;
};

function toLabel(text: string): string {
  return text
    .split("_")
    .map(function (part) {
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

function formatMetric(value: number): string {
  if (!Number.isFinite(value)) {
    return "N/A";
  }
  return value.toFixed(3);
}

function buildDefaultManualInputs(): ManualInputMatrix {
  const baseBand: Record<Band, number> = {
    delta: 2.4,
    theta: 2.1,
    low_alpha: 1.85,
    high_alpha: 1.75,
    alpha: 1.8,
    low_beta: 1.45,
    beta: 1.35,
    high_beta: 1.2,
  };

  const electrodeOffset: Record<Electrode, number> = {
    cz: 0,
    f3: 0.08,
    f4: 0.06,
    fz: 0.04,
    o1: -0.03,
  };

  const conditionOffset: Record<Condition, number> = {
    eo: 0.05,
    ec: 0,
  };

  const matrix = {} as ManualInputMatrix;

  for (const condition of CONDITIONS) {
    matrix[condition] = {} as Record<Electrode, Record<Band, string>>;
    for (const electrode of ELECTRODES) {
      matrix[condition][electrode] = {} as Record<Band, string>;
      for (const band of BANDS) {
        const value =
          baseBand[band] + electrodeOffset[electrode] + conditionOffset[condition];
        matrix[condition][electrode][band] = value.toFixed(3);
      }
    }
  }

  return matrix;
}

function toNumericManualInputs(inputs: ManualInputMatrix): ManualNumericMatrix {
  const matrix = {} as ManualNumericMatrix;

  for (const condition of CONDITIONS) {
    matrix[condition] = {} as Record<Electrode, Record<Band, number>>;
    for (const electrode of ELECTRODES) {
      matrix[condition][electrode] = {} as Record<Band, number>;
      for (const band of BANDS) {
        const rawValue = inputs[condition][electrode][band];
        const numericValue = Number(rawValue);
        if (!Number.isFinite(numericValue)) {
          throw new Error(
            "Please enter a valid number for " +
              condition.toUpperCase() +
              " " +
              electrode.toUpperCase() +
              " " +
              toLabel(band) +
              "."
          );
        }
        matrix[condition][electrode][band] = numericValue;
      }
    }
  }

  return matrix;
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();

    if (typeof payload?.detail === "string" && payload.detail.length > 0) {
      return payload.detail;
    }

    if (Array.isArray(payload?.detail) && payload.detail.length > 0) {
      return "Validation failed. Please check your input values.";
    }
  } catch {
    return "Request failed with status " + response.status + ".";
  }

  return "Request failed with status " + response.status + ".";
}

function HeroCard(props: HeroCardProps) {
  return (
    <section className="card hero-card">
      <p className="eyebrow">NeuroScope</p>
      <h1>NeuroScope Demo UI for Saved EEG Benchmark Artifacts</h1>
      <p className="subtitle">
        Lightweight demo interface for trying CSV, manual, and built-in example
        prediction workflows on top of the saved backend artifacts.
      </p>
      <p className="disclaimer">
        This UI is an optional demo layer. The recommended browser path
        uses <code>{props.apiBase}</code>, while the benchmark remains the main
        validated contribution.
      </p>
    </section>
  );
}

function WorkflowGrid(props: WorkflowGridProps) {
  return (
    <section className="workflow-grid">
      <article className="card workflow-card recommended-card">
        <div className="section-head">
          <h2>Upload CSV</h2>
          <span className="badge">Recommended</span>
        </div>
        <p className="section-note">
          Best for demonstrating the batch prediction path when you already have
          anchor EEG features in tabular format.
        </p>
        <form onSubmit={props.onCsvSubmit} className="stack-gap">
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={function (event: ChangeEvent<HTMLInputElement>) {
              props.onCsvFileSelect(
                event.target.files && event.target.files[0]
                  ? event.target.files[0]
                  : null
              );
            }}
          />
          <button
            type="submit"
            disabled={props.activeWorkflow !== null}
            className="btn btn-primary"
          >
            {props.activeWorkflow === "csv" ? "Uploading CSV..." : "Predict from CSV"}
          </button>
        </form>
      </article>

      <article className="card workflow-card">
        <div className="section-head">
          <h2>Upload EEG Report PDF</h2>
          <span className="badge soft">Future Extension</span>
        </div>
        <p className="section-note">
          Exploratory extension for future report parsing. The current implementation
          returns a structured acknowledgement response rather than a model prediction.
        </p>
        <form onSubmit={props.onPdfSubmit} className="stack-gap">
          <input
            type="file"
            accept="application/pdf,.pdf"
            onChange={function (event: ChangeEvent<HTMLInputElement>) {
              props.onPdfFileSelect(
                event.target.files && event.target.files[0]
                  ? event.target.files[0]
                  : null
              );
            }}
          />
          <button
            type="submit"
            disabled={props.activeWorkflow !== null}
            className="btn btn-secondary"
          >
            {props.activeWorkflow === "pdf" ? "Uploading PDF..." : "Try PDF Extension"}
          </button>
        </form>
      </article>

      <article className="card workflow-card">
        <div className="section-head">
          <h2>Try Demo Example</h2>
        </div>
        <p className="section-note">
          Quick way to show the API and UI path using built-in anchor EEG values
          from the backend.
        </p>
        <button
          type="button"
          disabled={props.activeWorkflow !== null}
          className="btn btn-secondary"
          onClick={props.onDemoClick}
        >
          {props.activeWorkflow === "demo" ? "Running Demo..." : "Run Demo Prediction"}
        </button>
      </article>
    </section>
  );
}

function OptionalManualEntryDemo(props: OptionalManualEntryDemoProps) {
  return (
    <details className="card advanced-card">
      <summary>Optional Manual Entry Demo (EO + EC)</summary>
      <p className="section-note advanced-note">
        Optional workflow for full control over anchor values.
        Inputs are grouped by condition and electrode.
      </p>
      <form onSubmit={props.onSubmit} className="stack-gap large-gap">
        <div className="manual-meta-grid">
          <label className="field">
            <span>Age</span>
            <input
              type="number"
              min="1"
              max="120"
              step="1"
              value={props.age}
              onChange={function (event: ChangeEvent<HTMLInputElement>) {
                props.onAgeChange(event.target.value);
              }}
            />
          </label>

          <label className="field">
            <span>Gender</span>
            <select
              value={props.gender}
              onChange={function (event: ChangeEvent<HTMLSelectElement>) {
                props.onGenderChange(event.target.value);
              }}
            >
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="1">Numeric 1</option>
              <option value="0">Numeric 0</option>
            </select>
          </label>
        </div>

        <div className="condition-wrapper">
          {CONDITIONS.map(function (condition) {
            return (
              <section key={condition} className="condition-block">
                <h3>{condition === "eo" ? "Eyes Open (EO)" : "Eyes Closed (EC)"}</h3>
                <div className="electrode-grid">
                  {ELECTRODES.map(function (electrode) {
                    return (
                      <fieldset key={electrode} className="electrode-card">
                        <legend>{electrode.toUpperCase()}</legend>
                        <div className="band-grid">
                          {BANDS.map(function (band) {
                            return (
                              <label key={band} className="field compact">
                                <span>{toLabel(band)}</span>
                                <input
                                  type="number"
                                  step="0.001"
                                  value={props.manualInputs[condition][electrode][band]}
                                  onChange={function (event: ChangeEvent<HTMLInputElement>) {
                                    props.onManualInputChange(
                                      condition,
                                      electrode,
                                      band,
                                      event.target.value
                                    );
                                  }}
                                />
                              </label>
                            );
                          })}
                        </div>
                      </fieldset>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>

        <button
          type="submit"
          disabled={props.activeWorkflow !== null}
          className="btn btn-secondary"
        >
          {props.activeWorkflow === "manual"
            ? "Submitting Manual Values..."
            : "Predict from Manual Entry"}
        </button>
      </form>
    </details>
  );
}

function MetricsGrid(props: { prediction: PredictionMetrics }) {
  return (
    <div className="metrics-grid">
      {TARGET_KEYS.map(function (key) {
        return (
          <article key={key} className="metric-card">
            <p className="metric-label">{toLabel(key)}</p>
            <p className="metric-value">{formatMetric(props.prediction[key])}</p>
          </article>
        );
      })}
    </div>
  );
}

function BatchResultsTable(props: { rows: CsvPredictionItem[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Row</th>
            {TARGET_KEYS.map(function (key) {
              return <th key={key}>{toLabel(key)}</th>;
            })}
          </tr>
        </thead>
        <tbody>
          {props.rows.map(function (item) {
            return (
              <tr key={item.row_index}>
                <td>{item.row_index}</td>
                {TARGET_KEYS.map(function (key) {
                  return <td key={key}>{formatMetric(item.predictions[key])}</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ResultsPanel(props: ResultsPanelProps) {
  return (
    <section className="card results-card">
      <div className="section-head">
        <h2>Results</h2>
      </div>

      {props.activeWorkflow !== null && (
        <div className="status loading-state">
          <span className="dot-loader" aria-hidden="true" />
          <p>Running {props.activeWorkflow.toUpperCase()} workflow...</p>
        </div>
      )}

      {props.error && (
        <div className="status error-state">
          <p>{props.error}</p>
        </div>
      )}

      {!props.activeWorkflow && !props.error && !props.result && (
        <p className="empty-state">
          No prediction yet. Start with the recommended CSV demo workflow or run
          the built-in demo.
        </p>
      )}

      {!props.activeWorkflow &&
        !props.error &&
        props.result &&
        props.result.mode === "single" && (
          <div className="result-block">
            <h3>{props.result.title}</h3>
            <MetricsGrid prediction={props.result.prediction} />
          </div>
        )}

      {!props.activeWorkflow &&
        !props.error &&
        props.result &&
        props.result.mode === "batch" && (
          <div className="result-block">
            <h3>{props.result.title}</h3>
            <BatchResultsTable rows={props.result.rows} />
          </div>
        )}

      {!props.activeWorkflow &&
        !props.error &&
        props.result &&
        props.result.mode === "info" && (
          <div className="result-block info-block">
            <h3>{props.result.title}</h3>
            <p>{props.result.message}</p>
            <p>{props.result.notes}</p>
          </div>
        )}
    </section>
  );
}

export default function HomePage() {
  const [age, setAge] = useState("29");
  const [gender, setGender] = useState("male");
  const [manualInputs, setManualInputs] = useState<ManualInputMatrix>(
    buildDefaultManualInputs
  );

  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [pdfFile, setPdfFile] = useState<File | null>(null);

  const [activeWorkflow, setActiveWorkflow] = useState<WorkflowKind | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ResultState | null>(null);

  function startWorkflow(workflow: WorkflowKind): void {
    setActiveWorkflow(workflow);
    setError(null);
  }

  function endWorkflow(): void {
    setActiveWorkflow(null);
  }

  function updateManualInput(
    condition: Condition,
    electrode: Electrode,
    band: Band,
    value: string
  ): void {
    setManualInputs(function (prev: ManualInputMatrix) {
      return {
        ...prev,
        [condition]: {
          ...prev[condition],
          [electrode]: {
            ...prev[condition][electrode],
            [band]: value,
          },
        },
      };
    });
  }

  async function handleCsvSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();

    if (!csvFile) {
      setError("Please choose a CSV file before submitting.");
      return;
    }

    startWorkflow("csv");

    try {
      const formData = new FormData();
      formData.append("file", csvFile);

      const response = await fetch(API_BASE + "/predict-from-csv", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const data = (await response.json()) as {
        rows: number;
        predictions: CsvPredictionItem[];
      };

      setResult({
        mode: "batch",
        title: "CSV Prediction Results",
        rows: data.predictions,
      });
    } catch (requestError) {
      if (requestError instanceof Error) {
        setError(requestError.message);
      } else {
        setError("CSV request failed unexpectedly.");
      }
    } finally {
      endWorkflow();
    }
  }

  async function handlePdfSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();

    if (!pdfFile) {
      setError("Please choose a PDF file before submitting.");
      return;
    }

    startWorkflow("pdf");

    try {
      const formData = new FormData();
      formData.append("file", pdfFile);

      const response = await fetch(API_BASE + "/predict-from-pdf", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const data = (await response.json()) as PdfResponse;

      setResult({
        mode: "info",
        title: "PDF Workflow Response",
        message: data.message,
        notes: data.notes,
      });
    } catch (requestError) {
      if (requestError instanceof Error) {
        setError(requestError.message);
      } else {
        setError("PDF request failed unexpectedly.");
      }
    } finally {
      endWorkflow();
    }
  }

  async function handleDemoClick(): Promise<void> {
    startWorkflow("demo");

    try {
      const response = await fetch(API_BASE + "/predict-demo", {
        method: "GET",
      });

      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const data = (await response.json()) as { predictions: PredictionMetrics };

      setResult({
        mode: "single",
        title: "Demo Prediction",
        prediction: data.predictions,
      });
    } catch (requestError) {
      if (requestError instanceof Error) {
        setError(requestError.message);
      } else {
        setError("Demo request failed unexpectedly.");
      }
    } finally {
      endWorkflow();
    }
  }

  async function handleManualSubmit(
    event: FormEvent<HTMLFormElement>
  ): Promise<void> {
    event.preventDefault();

    const numericAge = Number(age);
    if (!Number.isFinite(numericAge) || numericAge <= 0) {
      setError("Please enter a valid age greater than 0.");
      return;
    }

    let numericInputs: ManualNumericMatrix;
    try {
      numericInputs = toNumericManualInputs(manualInputs);
    } catch (conversionError) {
      if (conversionError instanceof Error) {
        setError(conversionError.message);
      } else {
        setError("Manual input validation failed.");
      }
      return;
    }

    startWorkflow("manual");

    try {
      const payload = {
        age: numericAge,
        gender,
        eo: numericInputs.eo,
        ec: numericInputs.ec,
      };

      const response = await fetch(API_BASE + "/predict-manual", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const data = (await response.json()) as { predictions: PredictionMetrics };

      setResult({
        mode: "single",
        title: "Manual Prediction",
        prediction: data.predictions,
      });
    } catch (requestError) {
      if (requestError instanceof Error) {
        setError(requestError.message);
      } else {
        setError("Manual request failed unexpectedly.");
      }
    } finally {
      endWorkflow();
    }
  }

  return (
    <main className="page-shell">
      <HeroCard apiBase={API_BASE} />
      <WorkflowGrid
        activeWorkflow={activeWorkflow}
        onCsvSubmit={handleCsvSubmit}
        onPdfSubmit={handlePdfSubmit}
        onDemoClick={handleDemoClick}
        onCsvFileSelect={setCsvFile}
        onPdfFileSelect={setPdfFile}
      />
      <OptionalManualEntryDemo
        age={age}
        gender={gender}
        manualInputs={manualInputs}
        activeWorkflow={activeWorkflow}
        onAgeChange={setAge}
        onGenderChange={setGender}
        onManualInputChange={updateManualInput}
        onSubmit={handleManualSubmit}
      />
      <ResultsPanel
        activeWorkflow={activeWorkflow}
        error={error}
        result={result}
      />
    </main>
  );
}
