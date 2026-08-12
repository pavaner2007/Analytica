import { useEffect, useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  Filter,
  Layers,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Table2,
  UploadCloud,
  Wand2,
} from 'lucide-react'
import {
  cleanDataset,
  exportUrl,
  fetchAudit,
  fetchChart,
  filterDataset,
  loadSample,
  trainModel,
  transformDataset,
  uploadDataset,
} from './api.js'

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Activity },
  { id: 'upload', label: 'Upload', icon: UploadCloud },
  { id: 'clean', label: 'Clean', icon: Wand2 },
  { id: 'transform', label: 'Transform', icon: RefreshCw },
  { id: 'filter', label: 'Filter', icon: Filter },
  { id: 'visualize', label: 'Visualize', icon: BarChart3 },
  { id: 'ml', label: 'ML Models', icon: Brain },
  { id: 'audit', label: 'Audit Trail', icon: ShieldCheck },
]

const chartTypes = ['histogram', 'scatter', 'bar', 'box']
const operators = [
  ['contains', 'Contains'],
  ['equals', 'Equals'],
  ['not_equals', 'Not Equals'],
  ['greater_than', 'Greater Than'],
  ['less_than', 'Less Than'],
  ['between', 'Between'],
]

function classNames(...values) {
  return values.filter(Boolean).join(' ')
}

function StatCard({ label, value, hint, icon: Icon }) {
  return (
    <div className="stat-card">
      <div>
        <p>{label}</p>
        <h3>{value}</h3>
        {hint && <span>{hint}</span>}
      </div>
      <div className="stat-icon"><Icon size={24} /></div>
    </div>
  )
}

function EmptyState({ onLoadSample }) {
  return (
    <div className="empty-state">
      <div className="orb orb-one" />
      <div className="orb orb-two" />
      <Sparkles size={44} />
      <h2>Start your data science workflow</h2>
      <p>Upload a CSV or Excel file, or open the sample dataset to explore cleaning, charts, and ML predictions.</p>
      <button className="primary-button" onClick={onLoadSample}>Load Sample Dataset</button>
    </div>
  )
}

function DataTable({ rows }) {
  const columns = useMemo(() => (rows?.length ? Object.keys(rows[0]) : []), [rows])
  if (!rows?.length) return <p className="muted">No preview rows available.</p>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((col) => <th key={col}>{col}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>{columns.map((col) => <td key={col}>{String(row[col] ?? '')}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ColumnSelector({ columns, selected, setSelected, numericOnly = false, numericColumns = [] }) {
  const availableColumns = numericOnly ? numericColumns : columns
  return (
    <div className="column-selector-container">
      <div className="bulk-selection-bar" style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
        <button
          type="button"
          className="ghost-button small"
          onClick={() => setSelected(availableColumns)}
          style={{ fontSize: '0.8rem', padding: '6px 12px' }}
        >
          Select All
        </button>
        <button
          type="button"
          className="ghost-button small"
          onClick={() => setSelected([])}
          style={{ fontSize: '0.8rem', padding: '6px 12px' }}
        >
          Clear Selection
        </button>
      </div>
      <div className="chip-grid">
        {availableColumns.map((col) => {
          const active = selected.includes(col)
          return (
            <button
              key={col}
              className={classNames('chip', active && 'chip-active')}
              onClick={() => setSelected(active ? selected.filter((item) => item !== col) : [...selected, col])}
              type="button"
            >
              {col}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function Dashboard({ profile, onLoadSample }) {
  if (!profile) return <EmptyState onLoadSample={onLoadSample} />
  return (
    <div className="page-grid">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Analytica</p>
          <h1>Interactive analytics, cleaning and ML in one React dashboard.</h1>
          <p>Current file: <strong>{profile.file_name}</strong></p>
        </div>
        <div className="quality-ring">
          <span>{profile.quality.score}%</span>
          <small>Quality</small>
        </div>
      </section>

      <div className="stats-grid">
        <StatCard label="Rows" value={profile.shape.rows.toLocaleString()} icon={Database} />
        <StatCard label="Columns" value={profile.shape.columns} icon={Layers} />
        <StatCard label="Missing" value={`${profile.quality.missing_pct}%`} icon={FileSpreadsheet} />
        <StatCard label="Duplicates" value={`${profile.quality.duplicate_pct}%`} icon={RefreshCw} />
      </div>

      <section className="panel two-col">
        <div>
          <h2>AI-style Insights</h2>
          <div className="insights">
            {profile.insights.map((item) => <div className="insight" key={item}><CheckCircle2 size={18} />{item}</div>)}
          </div>
        </div>
        <div>
          <h2>Column Health</h2>
          <div className="column-list">
            {profile.columns.slice(0, 8).map((col) => (
              <div className="column-row" key={col.name}>
                <span>{col.name}</span>
                <small>{col.dtype} · {col.missing_pct}% missing</small>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="section-title">
          <h2>Dataset Preview</h2>
          <span>First 50 rows</span>
        </div>
        <DataTable rows={profile.preview} />
      </section>
    </div>
  )
}

function UploadPage({ onProfileUpdate, setMessage, startAction, busy }) {
  const [file, setFile] = useState(null)

  async function handleUpload(event) {
    event.preventDefault()
    if (!file) {
      setMessage({ type: 'error', text: 'Please choose a CSV or Excel file first.' })
      return
    }
    await startAction(async () => {
      const data = await uploadDataset(file)
      onProfileUpdate(data)
      setMessage({ type: 'success', text: 'Dataset uploaded successfully.' })
    })
  }

  async function handleSample() {
    await startAction(async () => {
      const data = await loadSample()
      onProfileUpdate(data)
      setMessage({ type: 'success', text: 'Sample dataset loaded.' })
    })
  }

  return (
    <section className="panel upload-panel">
      <div className="upload-box">
        <UploadCloud size={54} />
        <h2>Upload dataset</h2>
        <p>Supports CSV, XLSX and XLS files. Your dataset is processed locally by the FastAPI backend.</p>
        <form onSubmit={handleUpload}>
          <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => setFile(event.target.files?.[0] || null)} />
          <div className="button-row">
            <button className="primary-button" disabled={busy} type="submit">Upload File</button>
            <button className="ghost-button" disabled={busy} type="button" onClick={handleSample}>Use Sample Data</button>
          </div>
        </form>
      </div>
    </section>
  )
}

function CleanPage({ profile, onProfileUpdate, setMessage, startAction, busy }) {
  const [action, setAction] = useState('fill_missing')
  const [method, setMethod] = useState('mean')
  const [selected, setSelected] = useState([])
  const [fillValue, setFillValue] = useState('Unknown')

  useEffect(() => {
    if (profile?.numeric_columns?.length && !selected.length) setSelected(profile.numeric_columns.slice(0, 2))
  }, [profile])

  // Handle action change to set appropriate default methods
  function handleActionChange(newAction) {
    setAction(newAction)
    if (newAction === 'fill_missing') {
      setMethod('mean')
    } else if (newAction === 'remove_outliers') {
      setMethod('iqr')
    }
  }

  if (!profile) return <EmptyHint />

  async function applyCleaning() {
    if (action !== 'drop_duplicates' && !selected.length) {
      setMessage({ type: 'error', text: 'Please select at least one column.' })
      return
    }
    await startAction(async () => {
      const payload = { action, method, columns: selected, fill_value: fillValue }
      const data = await cleanDataset(profile.session_id, payload)
      onProfileUpdate(data)
      setMessage({ type: 'success', text: 'Cleaning operation applied.' })
    })
  }

  const showMethod = action === 'fill_missing' || action === 'remove_outliers'
  const showFillValue = action === 'fill_missing' && method === 'constant'
  const showColumns = action !== 'drop_duplicates'

  return (
    <section className="panel form-panel">
      <div className="section-title">
        <h2>Data Cleaning</h2>
        <span>Fix missing values, duplicates and outliers</span>
      </div>
      <div className="form-grid">
        <label>Action
          <select value={action} onChange={(e) => handleActionChange(e.target.value)}>
            <option value="fill_missing">Fill Missing Values</option>
            <option value="drop_missing">Drop Missing Rows</option>
            <option value="drop_duplicates">Remove Duplicates</option>
            <option value="remove_outliers">Remove Outliers</option>
          </select>
        </label>
        {showMethod && (
          <label>Method
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              {action === 'fill_missing' ? (
                <>
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                  <option value="mode">Mode</option>
                  <option value="constant">Constant</option>
                </>
              ) : (
                <>
                  <option value="iqr">IQR</option>
                  <option value="zscore">Z-Score</option>
                </>
              )}
            </select>
          </label>
        )}
        {showFillValue && (
          <label>Constant Fill Value
            <input value={fillValue} onChange={(e) => setFillValue(e.target.value)} />
          </label>
        )}
      </div>
      {showColumns && (
        <>
          <h3>Select columns</h3>
          <ColumnSelector columns={profile.column_names} selected={selected} setSelected={setSelected} />
        </>
      )}
      <button className="primary-button" disabled={busy} onClick={applyCleaning}>Apply Cleaning</button>
    </section>
  )
}

function TransformPage({ profile, onProfileUpdate, setMessage, startAction, busy }) {
  const [transformType, setTransformType] = useState('log')
  const [selected, setSelected] = useState([])

  useEffect(() => {
    if (profile?.numeric_columns?.length && !selected.length) {
      setSelected([profile.numeric_columns[0]])
    }
  }, [profile])

  if (!profile) return <EmptyHint />

  async function applyTransformation() {
    if (!selected.length) {
      setMessage({ type: 'error', text: 'Please select at least one numeric column to transform.' })
      return
    }
    await startAction(async () => {
      const payload = { columns: selected, transform_type: transformType }
      const data = await transformDataset(profile.session_id, payload)
      onProfileUpdate(data)
      setMessage({ type: 'success', text: 'Transformation applied successfully.' })
    })
  }

  return (
    <section className="panel form-panel">
      <div className="section-title">
        <h2>Data Transformation</h2>
        <span>Apply math and scaling functions to numeric columns</span>
      </div>
      <div className="form-grid">
        <label>Transformation Type
          <select value={transformType} onChange={(e) => setTransformType(e.target.value)}>
            <option value="log">Log Transformation (log1p)</option>
            <option value="sqrt">Square Root (sqrt)</option>
            <option value="square">Square (x²)</option>
            <option value="standardize">Standardize (Z-score Scaling)</option>
            <option value="normalize">Normalize (Min-Max 0-1 Scaling)</option>
          </select>
        </label>
      </div>
      <h3>Select numeric columns to transform</h3>
      <ColumnSelector 
        columns={profile.column_names} 
        selected={selected} 
        setSelected={setSelected} 
        numericOnly={true} 
        numericColumns={profile.numeric_columns} 
      />
      <button className="primary-button" disabled={busy} onClick={applyTransformation}>Apply Transformation</button>
    </section>
  )
}

function FilterPage({ profile, onProfileUpdate, setMessage, startAction, busy }) {
  const [column, setColumn] = useState('')
  const [operator, setOperator] = useState('contains')
  const [value, setValue] = useState('')
  const [value2, setValue2] = useState('')

  useEffect(() => {
    if (profile?.column_names?.length) setColumn(profile.column_names[0])
  }, [profile])

  if (!profile) return <EmptyHint />

  async function applyFilter() {
    await startAction(async () => {
      const data = await filterDataset(profile.session_id, { column, operator, value, value2 })
      onProfileUpdate(data)
      setMessage({ type: 'success', text: 'Dataset filtered successfully.' })
    })
  }

  return (
    <section className="panel form-panel">
      <div className="section-title">
        <h2>Filtering & Selection</h2>
        <span>Keep only the records you need</span>
      </div>
      <div className="form-grid">
        <label>Column
          <select value={column} onChange={(e) => setColumn(e.target.value)}>
            {profile.column_names.map((col) => <option key={col} value={col}>{col}</option>)}
          </select>
        </label>
        <label>Operator
          <select value={operator} onChange={(e) => setOperator(e.target.value)}>
            {operators.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label>Value
          <input value={value} onChange={(e) => setValue(e.target.value)} placeholder="Enter filter value" />
        </label>
        {operator === 'between' && (
          <label>Second Value
            <input value={value2} onChange={(e) => setValue2(e.target.value)} placeholder="Enter second value" />
          </label>
        )}
      </div>
      <button className="primary-button" disabled={busy} onClick={applyFilter}>Apply Filter</button>
      <div className="preview-space"><DataTable rows={profile.preview} /></div>
    </section>
  )
}

function VisualizePage({ profile, startAction, setMessage }) {
  const [chartType, setChartType] = useState('histogram')
  const [x, setX] = useState('')
  const [y, setY] = useState('')
  const [color, setColor] = useState('')
  const [figure, setFigure] = useState(null)

  useEffect(() => {
    if (profile?.column_names?.length) {
      setX(profile.numeric_columns[0] || profile.column_names[0])
      setY(profile.numeric_columns[1] || profile.numeric_columns[0] || '')
    }
  }, [profile])

  if (!profile) return <EmptyHint />

  async function generateChart() {
    await startAction(async () => {
      const fig = await fetchChart(profile.session_id, { chart_type: chartType, x, y, color })
      setFigure(fig)
      setMessage({ type: 'success', text: 'Visualization generated.' })
    })
  }

  return (
    <section className="panel form-panel">
      <div className="section-title">
        <h2>Visualizations</h2>
        <span>Interactive Plotly charts</span>
      </div>
      <div className="form-grid">
        <label>Chart Type
          <select value={chartType} onChange={(e) => setChartType(e.target.value)}>
            {chartTypes.map((type) => <option key={type} value={type}>{type}</option>)}
          </select>
        </label>
        <label>X Column
          <select value={x} onChange={(e) => setX(e.target.value)}>
            {profile.column_names.map((col) => <option key={col} value={col}>{col}</option>)}
          </select>
        </label>
        <label>Y Column
          <select value={y} onChange={(e) => setY(e.target.value)}>
            <option value="">Auto</option>
            {profile.column_names.map((col) => <option key={col} value={col}>{col}</option>)}
          </select>
        </label>
        <label>Color Group
          <select value={color} onChange={(e) => setColor(e.target.value)}>
            <option value="">None</option>
            {profile.column_names.map((col) => <option key={col} value={col}>{col}</option>)}
          </select>
        </label>
      </div>
      <button className="primary-button" onClick={generateChart}>Generate Chart</button>
      <div className="chart-card">
        {figure ? (
          <Plot data={figure.data} layout={{ ...figure.layout, autosize: true }} config={{ responsive: true, displaylogo: false }} className="plot" />
        ) : (
          <div className="chart-placeholder"><BarChart3 size={48} /><p>Generate a chart to preview it here.</p></div>
        )}
      </div>
    </section>
  )
}

function MLPage({ profile, startAction, setMessage, busy }) {
  const [target, setTarget] = useState('')
  const [features, setFeatures] = useState([])
  const [modelType, setModelType] = useState('auto')
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (profile?.column_names?.length) {
      const defaultTarget = profile.numeric_columns[0] || profile.column_names[profile.column_names.length - 1]
      setTarget(defaultTarget)
      setFeatures(profile.column_names.filter((col) => col !== defaultTarget).slice(0, 6))
    }
  }, [profile])

  if (!profile) return <EmptyHint />

  async function train() {
    await startAction(async () => {
      const data = await trainModel(profile.session_id, { target, features, model_type: modelType })
      setResult(data)
      setMessage({ type: 'success', text: 'ML model trained successfully.' })
    })
  }

  return (
    <section className="panel form-panel">
      <div className="section-title">
        <h2>ML Predictions</h2>
        <span>Train a quick baseline model</span>
      </div>
      <div className="form-grid">
        <label>Target Column
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            {profile.column_names.map((col) => <option key={col} value={col}>{col}</option>)}
          </select>
        </label>
        <label>Model
          <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
            <option value="auto">Auto Select</option>
            <option value="random_forest">Random Forest</option>
            <option value="linear_regression">Linear Regression</option>
            <option value="logistic_regression">Logistic Regression</option>
          </select>
        </label>
      </div>
      <h3>Feature columns</h3>
      <ColumnSelector columns={profile.column_names.filter((col) => col !== target)} selected={features} setSelected={setFeatures} />
      <button className="primary-button" disabled={busy} onClick={train}>Train Model</button>
      {result && (
        <div className="model-result">
          <div className="result-header">
            <Brain size={28} />
            <div>
              <h3>{result.model}</h3>
              <p>{result.metrics.task} · {result.rows_used} rows used</p>
            </div>
          </div>
          <div className="metric-pills">
            {Object.entries(result.metrics).filter(([key]) => key !== 'task').map(([key, value]) => (
              <span key={key}>{key.replace('_', ' ')}: <strong>{value}</strong></span>
            ))}
          </div>
          <DataTable rows={result.sample_predictions} />
        </div>
      )}
    </section>
  )
}

function AuditPage({ audit, refreshAudit }) {
  return (
    <section className="panel">
      <div className="section-title">
        <h2>Audit Trail</h2>
        <button className="ghost-button small" onClick={refreshAudit}>Refresh</button>
      </div>
      {audit.length ? (
        <div className="audit-list">
          {audit.map((item, index) => (
            <div className="audit-item" key={`${item.timestamp}-${index}`}>
              <span>{item.action}</span>
              <p>{item.details}</p>
              <small>{item.timestamp}</small>
            </div>
          ))}
        </div>
      ) : <p className="muted">No audit logs yet.</p>}
    </section>
  )
}

function EmptyHint() {
  return (
    <section className="panel empty-hint">
      <Database size={40} />
      <h2>No active dataset</h2>
      <p>Upload a dataset or load the sample dataset first.</p>
    </section>
  )
}

export default function App() {
  const [activePage, setActivePage] = useState('dashboard')
  const [profile, setProfile] = useState(null)
  const [message, setMessage] = useState(null)
  const [busy, setBusy] = useState(false)
  const [audit, setAudit] = useState([])

  async function startAction(action) {
    setBusy(true)
    setMessage(null)
    try {
      await action()
      await refreshAudit()
    } catch (error) {
      setMessage({ type: 'error', text: error.message })
    } finally {
      setBusy(false)
    }
  }

  async function refreshAudit() {
    try {
      const data = await fetchAudit()
      setAudit(data.logs || [])
    } catch (_) {}
  }

  async function handleLoadSample() {
    await startAction(async () => {
      const data = await loadSample()
      setProfile(data)
      setMessage({ type: 'success', text: 'Sample dataset loaded successfully.' })
    })
  }

  useEffect(() => {
    refreshAudit()
  }, [])

  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => {
        setMessage(null)
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [message])

  const currentNav = navItems.find((item) => item.id === activePage)

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo"><Sparkles size={24} /></div>
          <div>
            <h2>Analytica</h2>
            <p>React + FastAPI</p>
          </div>
        </div>
        <nav>
          {navItems.map(({ id, label, icon: Icon }) => (
            <button key={id} className={classNames(activePage === id && 'active')} onClick={() => setActivePage(id)}>
              <Icon size={19} />{label}
            </button>
          ))}
        </nav>
        <div className="sidebar-card">
          <p>Current Session</p>
          <strong>{profile ? profile.shape.rows.toLocaleString() : 0}</strong>
          <span>Rows loaded</span>
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">{currentNav?.label}</p>
            <h1>{profile ? profile.file_name : 'Analytica'}</h1>
          </div>
          <div className="topbar-actions">
            {busy && <span className="loading"><Loader2 className="spin" size={18} />Processing</span>}
            {profile && <a className="download-button" href={exportUrl(profile.session_id)}><Download size={18} />Export CSV</a>}
          </div>
        </header>

        {message && <div className={classNames('toast', message.type)}>{message.text}</div>}

        <div className="page-fade-in" key={activePage}>
          {activePage === 'dashboard' && <Dashboard profile={profile} onLoadSample={handleLoadSample} />}
          {activePage === 'upload' && <UploadPage onProfileUpdate={setProfile} setMessage={setMessage} startAction={startAction} busy={busy} />}
          {activePage === 'clean' && <CleanPage profile={profile} onProfileUpdate={setProfile} setMessage={setMessage} startAction={startAction} busy={busy} />}
          {activePage === 'transform' && <TransformPage profile={profile} onProfileUpdate={setProfile} setMessage={setMessage} startAction={startAction} busy={busy} />}
          {activePage === 'filter' && <FilterPage profile={profile} onProfileUpdate={setProfile} setMessage={setMessage} startAction={startAction} busy={busy} />}
          {activePage === 'visualize' && <VisualizePage profile={profile} setMessage={setMessage} startAction={startAction} />}
          {activePage === 'ml' && <MLPage profile={profile} setMessage={setMessage} startAction={startAction} busy={busy} />}
          {activePage === 'audit' && <AuditPage audit={audit} refreshAudit={refreshAudit} />}
        </div>
      </main>
    </div>
  )
}
