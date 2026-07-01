import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Boxes,
  Brain,
  Database,
  GitBranch,
  Layers,
  RefreshCw,
  Upload,
  Zap
} from "lucide-react";
import "./styles.css";

const API_BASE = "http://127.0.0.1:8000/api";

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`);

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

async function apiPost(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

async function uploadDataset(file, name) {
  const formData = new FormData();
  formData.append("file", file);

  if (name) {
    formData.append("name", name);
  }

  const response = await fetch(`${API_BASE}/datasets/upload`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

function StatCard({ icon: Icon, label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">
        <Icon size={20} />
      </div>
      <div>
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
      </div>
    </div>
  );
}

function Section({ title, children, action }) {
  return (
    <section className="section">
      <div className="section-header">
        <h2>{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

function EmptyState({ message }) {
  return <div className="empty-state">{message}</div>;
}

function JsonBlock({ value }) {
  return (
    <pre className="json-block">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function DatasetUpload({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");
  const [status, setStatus] = useState("");

  async function handleUpload(event) {
    event.preventDefault();

    if (!file) {
      setStatus("Select a CSV first.");
      return;
    }

    try {
      setStatus("Uploading...");
      await uploadDataset(file, name);
      setStatus("Dataset uploaded.");
      setFile(null);
      setName("");
      onUploaded();
    } catch (error) {
      setStatus(error.message);
    }
  }

  return (
    <form className="inline-form" onSubmit={handleUpload}>
      <input
        type="file"
        accept=".csv"
        onChange={(event) => setFile(event.target.files?.[0] || null)}
      />
      <input
        placeholder="dataset name"
        value={name}
        onChange={(event) => setName(event.target.value)}
      />
      <button type="submit">
        <Upload size={16} />
        Upload CSV
      </button>
      {status && <span className="form-status">{status}</span>}
    </form>
  );
}

function DatasetTable({ datasets, onSelect }) {
  if (!datasets.length) {
    return <EmptyState message="No datasets registered yet." />;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Version</th>
            <th>Rows</th>
            <th>Columns</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((dataset) => (
            <tr key={dataset.id} onClick={() => onSelect(dataset.id)}>
              <td>{dataset.id}</td>
              <td>{dataset.name}</td>
              <td>v{dataset.version}</td>
              <td>{dataset.rows}</td>
              <td>{dataset.columns}</td>
              <td>{new Date(dataset.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FeatureTable({ features }) {
  if (!features.length) {
    return <EmptyState message="No feature definitions found." />;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Version</th>
            <th>Dataset</th>
            <th>Kind</th>
            <th>Transform</th>
            <th>Aggregation</th>
          </tr>
        </thead>
        <tbody>
          {features.map((feature) => (
            <tr key={feature.id}>
              <td>{feature.id}</td>
              <td>{feature.name}</td>
              <td>v{feature.version}</td>
              <td>{feature.dataset_id}</td>
              <td>{feature.feature_kind}</td>
              <td>{feature.transformation}</td>
              <td>{feature.aggregation_function || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MaterializationTable({ materializations }) {
  if (!materializations.length) {
    return <EmptyState message="No materialized feature tables found." />;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Dataset</th>
            <th>Rows</th>
            <th>Columns</th>
            <th>File</th>
          </tr>
        </thead>
        <tbody>
          {materializations.map((item) => (
            <tr key={item.id}>
              <td>{item.id}</td>
              <td>{item.name}</td>
              <td>{item.dataset_id}</td>
              <td>{item.rows}</td>
              <td>{item.columns}</td>
              <td>{item.stored_filename}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelTable({ models }) {
  if (!models.length) {
    return <EmptyState message="No trained models registered yet." />;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Algorithm</th>
            <th>Type</th>
            <th>Materialization</th>
            <th>Label</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.id}>
              <td>{model.id}</td>
              <td>{model.name}</td>
              <td>{model.algorithm}</td>
              <td>{model.problem_type}</td>
              <td>{model.materialization_id}</td>
              <td>{model.label_column}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DriftTable({ reports }) {
  if (!reports.length) {
    return <EmptyState message="No drift reports generated yet." />;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Reference</th>
            <th>Current</th>
            <th>Score</th>
            <th>Level</th>
          </tr>
        </thead>
        <tbody>
          {reports.map((report) => (
            <tr key={report.id}>
              <td>{report.id}</td>
              <td>{report.name}</td>
              <td>{report.reference_materialization_id}</td>
              <td>{report.current_materialization_id}</td>
              <td>{report.overall_drift_score}</td>
              <td>
                <span className={`pill pill-${report.drift_level}`}>
                  {report.drift_level}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CreateFeaturePanel({ datasets, onDone }) {
  const [form, setForm] = useState({
    name: "amount_log",
    dataset_id: "",
    description: "Log-transformed transaction amount.",
    entity_column: "user_id",
    source_column: "amount",
    feature_kind: "column",
    transformation: "log1p",
    aggregation_function: "",
    output_dtype: "float"
  });
  const [status, setStatus] = useState("");

  async function submit(event) {
    event.preventDefault();

    const payload = {
      ...form,
      dataset_id: Number(form.dataset_id),
      aggregation_function:
        form.feature_kind === "aggregate" ? form.aggregation_function : null
    };

    try {
      setStatus("Creating feature...");
      await apiPost("/features", payload);
      setStatus("Feature created.");
      onDone();
    } catch (error) {
      setStatus(error.message);
    }
  }

  return (
    <form className="panel-form" onSubmit={submit}>
      <div className="form-grid">
        <input
          placeholder="feature name"
          value={form.name}
          onChange={(event) => setForm({ ...form, name: event.target.value })}
        />
        <select
          value={form.dataset_id}
          onChange={(event) => setForm({ ...form, dataset_id: event.target.value })}
        >
          <option value="">Select dataset</option>
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.id} — {dataset.name}
            </option>
          ))}
        </select>
        <input
          placeholder="entity_column"
          value={form.entity_column}
          onChange={(event) =>
            setForm({ ...form, entity_column: event.target.value })
          }
        />
        <input
          placeholder="source_column"
          value={form.source_column}
          onChange={(event) =>
            setForm({ ...form, source_column: event.target.value })
          }
        />
        <select
          value={form.feature_kind}
          onChange={(event) =>
            setForm({ ...form, feature_kind: event.target.value })
          }
        >
          <option value="column">column</option>
          <option value="aggregate">aggregate</option>
        </select>
        <select
          value={form.transformation}
          onChange={(event) =>
            setForm({ ...form, transformation: event.target.value })
          }
        >
          <option value="identity">identity</option>
          <option value="log1p">log1p</option>
          <option value="zscore">zscore</option>
          <option value="minmax">minmax</option>
          <option value="abs">abs</option>
          <option value="square">square</option>
        </select>
        {form.feature_kind === "aggregate" && (
          <select
            value={form.aggregation_function}
            onChange={(event) =>
              setForm({ ...form, aggregation_function: event.target.value })
            }
          >
            <option value="">aggregation</option>
            <option value="count">count</option>
            <option value="mean">mean</option>
            <option value="sum">sum</option>
            <option value="min">min</option>
            <option value="max">max</option>
            <option value="nunique">nunique</option>
          </select>
        )}
      </div>
      <button type="submit">Create Feature</button>
      {status && <div className="form-status block">{status}</div>}
    </form>
  );
}

function ActionPanel({ datasets, materializations, models, refreshAll }) {
  const [materializeForm, setMaterializeForm] = useState({
    dataset_id: "",
    name: "fraud_training_features",
    label_column: "is_fraud"
  });
  const [trainForm, setTrainForm] = useState({
    materialization_id: "",
    name: "fraud_detection_rf",
    label_column: "is_fraud",
    algorithm: "random_forest",
    problem_type: "classification"
  });
  const [onlineForm, setOnlineForm] = useState({
    materialization_id: "",
    entity_column: "user_id"
  });
  const [driftForm, setDriftForm] = useState({
    reference_materialization_id: "",
    current_materialization_id: "",
    name: "feature_drift_report"
  });
  const [output, setOutput] = useState(null);

  async function createMaterialization(event) {
    event.preventDefault();

    const result = await apiPost("/materializations", {
      dataset_id: Number(materializeForm.dataset_id),
      name: materializeForm.name,
      feature_ids: null,
      label_column: materializeForm.label_column
    });

    setOutput(result);
    refreshAll();
  }

  async function trainModel(event) {
    event.preventDefault();

    const result = await apiPost("/models/train", {
      materialization_id: Number(trainForm.materialization_id),
      name: trainForm.name,
      label_column: trainForm.label_column,
      algorithm: trainForm.algorithm,
      problem_type: trainForm.problem_type,
      test_size: 0.3,
      random_state: 42
    });

    setOutput(result);
    refreshAll();
  }

  async function pushOnline(event) {
    event.preventDefault();

    const result = await apiPost("/online-store/materialize", {
      materialization_id: Number(onlineForm.materialization_id),
      entity_column: onlineForm.entity_column,
      deduplication_strategy: "last"
    });

    setOutput(result);
    refreshAll();
  }

  async function createDriftReport(event) {
    event.preventDefault();

    const result = await apiPost("/drift/reports", {
      reference_materialization_id: Number(driftForm.reference_materialization_id),
      current_materialization_id: Number(driftForm.current_materialization_id),
      name: driftForm.name,
      feature_columns: null
    });

    setOutput(result);
    refreshAll();
  }

  return (
    <Section title="Operations">
      <div className="action-grid">
        <form className="action-card" onSubmit={createMaterialization}>
          <h3>Materialize Features</h3>
          <select
            value={materializeForm.dataset_id}
            onChange={(event) =>
              setMaterializeForm({
                ...materializeForm,
                dataset_id: event.target.value
              })
            }
          >
            <option value="">Select dataset</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.id} — {dataset.name}
              </option>
            ))}
          </select>
          <input
            value={materializeForm.name}
            onChange={(event) =>
              setMaterializeForm({ ...materializeForm, name: event.target.value })
            }
          />
          <input
            value={materializeForm.label_column}
            onChange={(event) =>
              setMaterializeForm({
                ...materializeForm,
                label_column: event.target.value
              })
            }
          />
          <button type="submit">Run</button>
        </form>

        <form className="action-card" onSubmit={trainModel}>
          <h3>Train Model</h3>
          <select
            value={trainForm.materialization_id}
            onChange={(event) =>
              setTrainForm({
                ...trainForm,
                materialization_id: event.target.value
              })
            }
          >
            <option value="">Select materialization</option>
            {materializations.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id} — {item.name}
              </option>
            ))}
          </select>
          <input
            value={trainForm.name}
            onChange={(event) =>
              setTrainForm({ ...trainForm, name: event.target.value })
            }
          />
          <select
            value={trainForm.algorithm}
            onChange={(event) =>
              setTrainForm({ ...trainForm, algorithm: event.target.value })
            }
          >
            <option value="random_forest">random_forest</option>
            <option value="logistic_regression">logistic_regression</option>
            <option value="xgboost">xgboost</option>
          </select>
          <button type="submit">Train</button>
        </form>

        <form className="action-card" onSubmit={pushOnline}>
          <h3>Push Online Store</h3>
          <select
            value={onlineForm.materialization_id}
            onChange={(event) =>
              setOnlineForm({
                ...onlineForm,
                materialization_id: event.target.value
              })
            }
          >
            <option value="">Select materialization</option>
            {materializations.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id} — {item.name}
              </option>
            ))}
          </select>
          <input
            value={onlineForm.entity_column}
            onChange={(event) =>
              setOnlineForm({ ...onlineForm, entity_column: event.target.value })
            }
          />
          <button type="submit">Push</button>
        </form>

        <form className="action-card" onSubmit={createDriftReport}>
          <h3>Create Drift Report</h3>
          <select
            value={driftForm.reference_materialization_id}
            onChange={(event) =>
              setDriftForm({
                ...driftForm,
                reference_materialization_id: event.target.value
              })
            }
          >
            <option value="">Reference</option>
            {materializations.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id} — {item.name}
              </option>
            ))}
          </select>
          <select
            value={driftForm.current_materialization_id}
            onChange={(event) =>
              setDriftForm({
                ...driftForm,
                current_materialization_id: event.target.value
              })
            }
          >
            <option value="">Current</option>
            {materializations.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id} — {item.name}
              </option>
            ))}
          </select>
          <button type="submit">Generate</button>
        </form>
      </div>

      {output && (
        <div className="output-panel">
          <h3>Latest Operation Output</h3>
          <JsonBlock value={output} />
        </div>
      )}
    </Section>
  );
}

function App() {
  const [datasets, setDatasets] = useState([]);
  const [features, setFeatures] = useState([]);
  const [materializations, setMaterializations] = useState([]);
  const [models, setModels] = useState([]);
  const [driftReports, setDriftReports] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function refreshAll() {
    setLoading(true);
    setError("");

    try {
      const [
        datasetData,
        featureData,
        materializationData,
        modelData,
        driftData
      ] = await Promise.all([
        apiGet("/datasets"),
        apiGet("/features"),
        apiGet("/materializations"),
        apiGet("/models"),
        apiGet("/drift/reports")
      ]);

      setDatasets(datasetData);
      setFeatures(featureData);
      setMaterializations(materializationData);
      setModels(modelData);
      setDriftReports(driftData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadDatasetPreview(datasetId) {
    setSelectedDataset(datasetId);

    try {
      const preview = await apiGet(`/datasets/${datasetId}/preview?limit=5`);
      setDatasetPreview(preview);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshAll();
  }, []);

  const latestModel = useMemo(() => models[0] || null, [models]);

  return (
    <div className="app">
      <header className="hero">
        <div>
          <div className="eyebrow">ML Infrastructure Project</div>
          <h1>FeatureForge</h1>
          <p>
            Lightweight feature store for dataset registration, reusable feature
            definitions, offline materialization, model training, online serving,
            and drift monitoring.
          </p>
        </div>
        <button className="refresh-button" onClick={refreshAll}>
          <RefreshCw size={16} />
          Refresh
        </button>
      </header>

      {error && <div className="error-box">{error}</div>}

      <div className="stats-grid">
        <StatCard icon={Database} label="Datasets" value={datasets.length} />
        <StatCard icon={GitBranch} label="Features" value={features.length} />
        <StatCard icon={Layers} label="Materializations" value={materializations.length} />
        <StatCard icon={Brain} label="Models" value={models.length} />
        <StatCard icon={Activity} label="Drift Reports" value={driftReports.length} />
        <StatCard icon={Zap} label="Latest Model" value={latestModel?.name || "None"} />
      </div>

      {loading && <div className="loading">Loading...</div>}

      <Section
        title="Datasets"
        action={<DatasetUpload onUploaded={refreshAll} />}
      >
        <DatasetTable datasets={datasets} onSelect={loadDatasetPreview} />
        {datasetPreview && (
          <div className="preview-panel">
            <h3>Dataset Preview: {selectedDataset}</h3>
            <JsonBlock value={datasetPreview.preview} />
          </div>
        )}
      </Section>

      <Section title="Create Feature">
        <CreateFeaturePanel datasets={datasets} onDone={refreshAll} />
      </Section>

      <ActionPanel
        datasets={datasets}
        materializations={materializations}
        models={models}
        refreshAll={refreshAll}
      />

      <Section title="Feature Registry">
        <FeatureTable features={features} />
      </Section>

      <Section title="Offline Materializations">
        <MaterializationTable materializations={materializations} />
      </Section>

      <Section title="Model Registry">
        <ModelTable models={models} />
      </Section>

      <Section title="Drift Reports">
        <DriftTable reports={driftReports} />
      </Section>

      <footer>
        <Boxes size={16} />
        FeatureForge MVP — backend + dashboard
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
