import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './App.css'

const API = 'http://localhost:8420'

interface Run {
  run_id: string
  config: string
  started: number
  finished: number | null
  status: string
  final_train_loss: number | null
  final_val_loss: number | null
}

interface LogEntry {
  step: number
  timestamp: number
  train_loss: number
  val_loss: number
  tokens_per_second: number
  tokens_seen: number
  sample_output: string
}

interface Checkpoint {
  name: string
  size_mb: number
  modified: number
}

function formatTime(ts: number | null): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString()
}

function formatDuration(started: number, finished: number | null): string {
  const end = finished || Date.now() / 1000
  const secs = end - started
  if (secs < 60) return `${Math.round(secs)}s`
  if (secs < 3600) return `${Math.round(secs / 60)}m`
  return `${(secs / 3600).toFixed(1)}h`
}

function parseConfig(configStr: string) {
  try { return JSON.parse(configStr) } catch { return {} }
}

function RunOverview({ runs, selectedRun, onSelectRun }: {
  runs: Run[]
  selectedRun: string | null
  onSelectRun: (id: string) => void
}) {
  return (
    <div className="panel">
      <h2>Training Runs</h2>
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Config</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Train Loss</th>
            <th>Val Loss</th>
          </tr>
        </thead>
        <tbody>
          {runs.map(run => {
            const cfg = parseConfig(run.config)
            return (
              <tr
                key={run.run_id}
                onClick={() => onSelectRun(run.run_id)}
                className={selectedRun === run.run_id ? 'selected' : ''}
              >
                <td className="mono">{run.run_id}</td>
                <td>{cfg.n_layer && cfg.n_head && cfg.n_embd ? `${cfg.n_layer}L/${cfg.n_head}H/${cfg.n_embd}E` : '—'}</td>
                <td>
                  <span className={`status ${run.status}`}>
                    {run.status === 'complete' ? '✓' : '●'} {run.status}
                  </span>
                </td>
                <td>{formatDuration(run.started, run.finished)}</td>
                <td className="mono">{run.final_train_loss?.toFixed(4) || '—'}</td>
                <td className="mono">{run.final_val_loss?.toFixed(4) || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function LossChart({ logs, title }: { logs: LogEntry[], title: string }) {
  if (logs.length === 0) return <div className="panel"><h2>{title}</h2><p>No log data yet.</p></div>

  return (
    <div className="panel">
      <h2>{title}</h2>
      <div style={{ width: '100%', height: 350 }}>
        <ResponsiveContainer>
          <LineChart data={logs}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="step" stroke="#888" />
            <YAxis stroke="#888" />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
            />
            <Legend />
            <Line type="monotone" dataKey="train_loss" stroke="#4fc3f7" name="Train Loss" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="val_loss" stroke="#ff8a65" name="Val Loss" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function ThroughputChart({ logs }: { logs: LogEntry[] }) {
  if (logs.length === 0) return null

  return (
    <div className="panel">
      <h2>Throughput</h2>
      <div style={{ width: '100%', height: 250 }}>
        <ResponsiveContainer>
          <LineChart data={logs}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="step" stroke="#888" />
            <YAxis stroke="#888" />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
              formatter={(value: number) => [`${Math.round(value).toLocaleString()} tok/s`, 'Throughput']}
            />
            <Line type="monotone" dataKey="tokens_per_second" stroke="#81c784" name="Tokens/sec" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function RunDetail({ run, logs }: { run: Run, logs: LogEntry[] }) {
  const cfg = parseConfig(run.config)

  return (
    <div className="panel">
      <h2>Run Detail: {run.run_id}</h2>
      <div className="detail-grid">
        <div className="detail-item">
          <span className="label">Model</span>
          <span className="value">{cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} embd</span>
        </div>
        <div className="detail-item">
          <span className="label">Block Size</span>
          <span className="value">{cfg.block_size}</span>
        </div>
        <div className="detail-item">
          <span className="label">Learning Rate</span>
          <span className="value">{cfg.learning_rate}</span>
        </div>
        <div className="detail-item">
          <span className="label">Batch Size</span>
          <span className="value">{cfg.batch_size}</span>
        </div>
        <div className="detail-item">
          <span className="label">Data</span>
          <span className="value">{cfg.data_file}</span>
        </div>
        <div className="detail-item">
          <span className="label">Started</span>
          <span className="value">{formatTime(run.started)}</span>
        </div>
      </div>
    </div>
  )
}

function Analysis({ run, logs }: { run: Run, logs: LogEntry[] }) {
  if (logs.length < 3) return null

  const cfg = parseConfig(run.config)
  const findings: { type: 'good' | 'warn' | 'bad' | 'info', text: string }[] = []

  // Find best val loss and where it occurred
  const valLosses = logs.filter(l => l.val_loss > 0)
  const bestVal = Math.min(...valLosses.map(l => l.val_loss))
  const bestValStep = valLosses.find(l => l.val_loss === bestVal)?.step || 0
  const lastVal = valLosses[valLosses.length - 1]
  const lastTrain = logs[logs.length - 1]
  const firstVal = valLosses[0]
  const maxStep = lastTrain?.step || 0

  // Did learning happen?
  if (lastVal && firstVal && lastVal.val_loss < firstVal.val_loss * 0.9) {
    findings.push({ type: 'good', text: `Model learned successfully. Val loss improved from ${firstVal.val_loss.toFixed(2)} to best of ${bestVal.toFixed(2)}.` })
  } else if (lastVal && firstVal) {
    findings.push({ type: 'bad', text: `Model did not improve much. Val loss went from ${firstVal.val_loss.toFixed(2)} to ${lastVal.val_loss.toFixed(2)}. Check data quality and learning rate.` })
  }

  // Overfitting check
  if (lastVal && lastTrain) {
    const gap = lastVal.val_loss - lastTrain.train_loss
    const valTrend = lastVal.val_loss - bestVal

    if (valTrend > 0.5 && gap > 0.5) {
      findings.push({ type: 'bad', text: `Severe overfitting detected. Val loss rose from ${bestVal.toFixed(2)} (step ${bestValStep}) to ${lastVal.val_loss.toFixed(2)}. Train/val gap is ${gap.toFixed(2)}. The model is memorizing training data instead of learning general patterns.` })

      // Diagnose cause
      const dataFile = cfg.data_file || ''
      if (dataFile.includes('input.txt') || dataFile.includes('shakespeare')) {
        findings.push({ type: 'info', text: `Likely cause: not enough training data. Shakespeare is only ~1MB. This model has enough capacity to memorize it. Solution: use a larger dataset (the React/TS corpus is 52MB, ~50x more data).` })
      } else {
        findings.push({ type: 'info', text: `Likely cause: model is too large for the dataset, or training ran too long. Try: more data, earlier stopping, more dropout, or a smaller model.` })
      }

      // Suggest best checkpoint
      findings.push({ type: 'info', text: `Best checkpoint was at step ${bestValStep} (val loss ${bestVal.toFixed(2)}). Use that checkpoint instead of the final one.` })

    } else if (valTrend > 0.1) {
      findings.push({ type: 'warn', text: `Mild overfitting. Val loss is ${valTrend.toFixed(2)} above its best. Train/val gap is ${gap.toFixed(2)}. Consider stopping earlier or adding dropout.` })
    } else {
      findings.push({ type: 'good', text: `No significant overfitting. Train/val gap is ${gap.toFixed(2)}, which is healthy.` })
    }
  }

  // Perplexity interpretation
  if (lastVal) {
    const ppl = Math.exp(bestVal)
    if (ppl < 10) {
      findings.push({ type: 'good', text: `Best perplexity: ${ppl.toFixed(1)}. The model is narrowing each prediction down to ~${Math.round(ppl)} plausible options. Reasonable for a small character model.` })
    } else if (ppl < 50) {
      findings.push({ type: 'warn', text: `Best perplexity: ${ppl.toFixed(1)}. The model is fairly uncertain — choosing between ~${Math.round(ppl)} options per character. More data or training may help.` })
    } else {
      findings.push({ type: 'bad', text: `Best perplexity: ${ppl.toFixed(1)}. Very high uncertainty. The model hasn't learned strong patterns. Check data quality and model configuration.` })
    }
  }

  // Throughput info
  const avgThroughput = logs.reduce((sum, l) => sum + (l.tokens_per_second || 0), 0) / logs.length
  if (avgThroughput > 0) {
    findings.push({ type: 'info', text: `Average throughput: ${Math.round(avgThroughput).toLocaleString()} tokens/sec. At this speed, training on 52MB of React/TS data would take roughly ${Math.round(52_000_000 / avgThroughput / 60)} minutes.` })
  }

  // Early stopping suggestion
  if (bestValStep < maxStep * 0.5) {
    findings.push({ type: 'warn', text: `Best val loss was at step ${bestValStep}, but training continued to step ${maxStep}. Over half the training was wasted. Next run should use early stopping or fewer steps.` })
  }

  // Next steps
  const suggestions: string[] = []
  if (lastVal && lastTrain && lastVal.val_loss - bestVal > 0.5) {
    suggestions.push('Use more training data to prevent overfitting')
    suggestions.push('Try adding dropout (0.1-0.2)')
    suggestions.push(`Use the step ${bestValStep} checkpoint instead of the final one`)
  }
  if (bestVal > 2.0) {
    suggestions.push('Train for more steps (loss is still high)')
    suggestions.push('Try a larger model if data is sufficient')
  }
  if (bestVal < 1.8 && !(lastVal && lastVal.val_loss - bestVal > 0.5)) {
    suggestions.push('Model looks healthy — ready to try on domain-specific data')
    suggestions.push('Consider scaling up model size with more data')
  }

  return (
    <div className="panel analysis">
      <h2>Analysis</h2>
      <div className="findings">
        {findings.map((f, i) => (
          <div key={i} className={`finding ${f.type}`}>
            <span className="finding-icon">
              {f.type === 'good' ? '✓' : f.type === 'warn' ? '!' : f.type === 'bad' ? '✗' : 'i'}
            </span>
            <span>{f.text}</span>
          </div>
        ))}
      </div>
      {suggestions.length > 0 && (
        <>
          <h3>Suggested Next Steps</h3>
          <ul className="suggestions">
            {suggestions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}

function Samples({ logs }: { logs: LogEntry[] }) {
  const samplesWithData = logs.filter(l => l.sample_output)
  if (samplesWithData.length === 0) return null

  return (
    <div className="panel">
      <h2>Generated Samples</h2>
      <div className="samples">
        {samplesWithData.map(log => (
          <div key={log.step} className="sample">
            <div className="sample-header">Step {log.step} — loss {log.val_loss.toFixed(4)}</div>
            <pre>{log.sample_output}</pre>
          </div>
        ))}
      </div>
    </div>
  )
}

function CompareView({ runs, allLogs }: {
  runs: Run[]
  allLogs: Record<string, LogEntry[]>
}) {
  if (runs.length < 2) return null

  const combinedData: Record<number, Record<string, number>> = {}
  for (const run of runs) {
    const logs = allLogs[run.run_id] || []
    for (const log of logs) {
      if (!combinedData[log.step]) combinedData[log.step] = { step: log.step }
      combinedData[log.step][`${run.run_id}_val`] = log.val_loss
    }
  }

  const data = Object.values(combinedData).sort((a, b) => a.step - b.step)
  const colors = ['#4fc3f7', '#ff8a65', '#81c784', '#ce93d8', '#fff176']

  return (
    <div className="panel">
      <h2>Compare: Val Loss</h2>
      <div style={{ width: '100%', height: 350 }}>
        <ResponsiveContainer>
          <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="step" stroke="#888" />
          <YAxis stroke="#888" />
          <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
          <Legend />
          {runs.map((run, i) => (
            <Line
              key={run.run_id}
              type="monotone"
              dataKey={`${run.run_id}_val`}
              stroke={colors[i % colors.length]}
              name={run.run_id}
              dot={false}
              strokeWidth={2}
            />
          ))}
        </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function TrainingPlanner() {
  const [corpusMB, setCorpusMB] = useState(52)
  const [modelParams, setModelParams] = useState(5)
  const [batchSize, setBatchSize] = useState(64)
  const [blockSize, setBlockSize] = useState(256)
  const [tokPerSec, setTokPerSec] = useState(34000)

  const corpusTokens = corpusMB * 1_000_000
  const modelTokens = modelParams * 1_000_000
  const chinchillaTokens = modelTokens * 20
  const tokensPerStep = batchSize * blockSize
  const passesThrough = Math.max(chinchillaTokens, corpusTokens) / corpusTokens
  const totalTokens = Math.max(chinchillaTokens, corpusTokens)
  const steps = Math.ceil(totalTokens / tokensPerStep)
  const timeSec = totalTokens / tokPerSec
  const timeMin = timeSec / 60
  const timeHrs = timeSec / 3600

  let riskLevel: 'low' | 'medium' | 'high'
  let riskText: string
  if (passesThrough <= 3) {
    riskLevel = 'low'
    riskText = `Low overfitting risk. Data seen ~${passesThrough.toFixed(1)}x — healthy range.`
  } else if (passesThrough <= 10) {
    riskLevel = 'medium'
    riskText = `Moderate overfitting risk. Data seen ~${passesThrough.toFixed(1)}x. Consider more data or dropout.`
  } else {
    riskLevel = 'high'
    riskText = `High overfitting risk! Data seen ~${passesThrough.toFixed(0)}x. Model will likely memorize. Use more data or a smaller model.`
  }

  let recommendation: string
  if (corpusTokens >= chinchillaTokens) {
    recommendation = `Your data (${corpusMB}MB) is sufficient for a ${modelParams}M model. You could even try a larger model.`
  } else {
    const idealMaxParams = Math.floor(corpusTokens / 20 / 1_000_000)
    recommendation = `Your data (${corpusMB}MB) is best suited for a model up to ~${idealMaxParams}M params. For ${modelParams}M params you ideally want ~${Math.ceil(chinchillaTokens / 1_000_000)}MB of data.`
  }

  return (
    <div className="panel">
      <h2>Training Planner</h2>
      <div className="planner-inputs">
        <label>
          <span>Corpus size (MB)</span>
          <input type="number" value={corpusMB} onChange={e => setCorpusMB(Number(e.target.value))} min={0.1} step={1} />
        </label>
        <label>
          <span>Model params (M)</span>
          <input type="number" value={modelParams} onChange={e => setModelParams(Number(e.target.value))} min={0.1} step={1} />
        </label>
        <label>
          <span>Batch size</span>
          <input type="number" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} min={1} />
        </label>
        <label>
          <span>Block size</span>
          <input type="number" value={blockSize} onChange={e => setBlockSize(Number(e.target.value))} min={32} />
        </label>
        <label>
          <span>Throughput (tok/s)</span>
          <input type="number" value={tokPerSec} onChange={e => setTokPerSec(Number(e.target.value))} min={100} step={1000} />
        </label>
      </div>

      <div className="planner-results">
        <div className="planner-row">
          <span className="planner-label">Corpus tokens</span>
          <span className="planner-value">{(corpusTokens / 1_000_000).toFixed(0)}M</span>
        </div>
        <div className="planner-row">
          <span className="planner-label">Chinchilla-optimal tokens</span>
          <span className="planner-value">{(chinchillaTokens / 1_000_000).toFixed(0)}M</span>
        </div>
        <div className="planner-row">
          <span className="planner-label">Tokens to train on</span>
          <span className="planner-value">{(totalTokens / 1_000_000).toFixed(0)}M</span>
        </div>
        <div className="planner-row">
          <span className="planner-label">Passes through data</span>
          <span className="planner-value">{passesThrough.toFixed(1)}x</span>
        </div>
        <div className="planner-row">
          <span className="planner-label">Training steps</span>
          <span className="planner-value">{steps.toLocaleString()}</span>
        </div>
        <div className="planner-row">
          <span className="planner-label">Estimated time</span>
          <span className="planner-value">
            {timeHrs >= 1 ? `${timeHrs.toFixed(1)} hours` : `${timeMin.toFixed(0)} minutes`}
          </span>
        </div>
      </div>

      <div className={`finding ${riskLevel === 'low' ? 'good' : riskLevel === 'medium' ? 'warn' : 'bad'}`} style={{ marginTop: 16 }}>
        <span className="finding-icon">{riskLevel === 'low' ? '✓' : riskLevel === 'medium' ? '!' : '✗'}</span>
        <span>{riskText}</span>
      </div>
      <div className="finding info" style={{ marginTop: 8 }}>
        <span className="finding-icon">i</span>
        <span>{recommendation}</span>
      </div>
    </div>
  )
}

function App() {
  const [runs, setRuns] = useState<Run[]>([])
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [allLogs, setAllLogs] = useState<Record<string, LogEntry[]>>({})
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [tab, setTab] = useState<'detail' | 'compare'>('detail')
  const [error, setError] = useState<string | null>(null)

  const fetchData = () => {
    fetch(`${API}/api/runs`)
      .then(r => r.json())
      .then(data => {
        setRuns(data)
        if (data.length > 0 && !selectedRun) {
          setSelectedRun(data[0].run_id)
        }
      })
      .catch(() => setError('Cannot connect to API. Run: python dashboard/api.py'))

    fetch(`${API}/api/checkpoints`)
      .then(r => r.json())
      .then(setCheckpoints)
      .catch(() => {})
  }

  useEffect(() => {
    fetchData()
  }, [])

  useEffect(() => {
    if (!selectedRun) return
    fetch(`${API}/api/runs/${selectedRun}/logs`)
      .then(r => r.json())
      .then(data => {
        setLogs(data)
        setAllLogs(prev => ({ ...prev, [selectedRun]: data }))
      })
      .catch(() => {})
  }, [selectedRun])

  // Load all run logs for comparison
  useEffect(() => {
    for (const run of runs) {
      if (!allLogs[run.run_id]) {
        fetch(`${API}/api/runs/${run.run_id}/logs`)
          .then(r => r.json())
          .then(data => {
            setAllLogs(prev => ({ ...prev, [run.run_id]: data }))
          })
          .catch(() => {})
      }
    }
  }, [runs])

  if (error) {
    return (
      <div className="app">
        <h1>TinyLLM Dashboard</h1>
        <div className="panel error">
          <p>{error}</p>
          <pre>cd tinyllm && source venv/bin/activate && uvicorn dashboard.api:app --port 8420</pre>
        </div>
      </div>
    )
  }

  const selectedRunData = runs.find(r => r.run_id === selectedRun)

  return (
    <div className="app">
      <header>
        <h1>TinyLLM Dashboard</h1>
        <div className="tabs">
          <button className={tab === 'detail' ? 'active' : ''} onClick={() => setTab('detail')}>Run Detail</button>
          <button className={tab === 'compare' ? 'active' : ''} onClick={() => setTab('compare')}>Compare Runs</button>
          <button onClick={() => { fetchData(); if (selectedRun) { fetch(`${API}/api/runs/${selectedRun}/logs`).then(r => r.json()).then(data => { setLogs(data); setAllLogs(prev => ({ ...prev, [selectedRun!]: data })) }) } }} className="refresh">Refresh</button>
        </div>
      </header>

      <RunOverview runs={runs} selectedRun={selectedRun} onSelectRun={setSelectedRun} />

      {tab === 'detail' && selectedRunData && (
        <>
          <RunDetail run={selectedRunData} logs={logs} />
          <Analysis run={selectedRunData} logs={logs} />
          <LossChart logs={logs} title={`Loss Curves: ${selectedRun}`} />
          <ThroughputChart logs={logs} />
          <Samples logs={logs} />
        </>
      )}

      {tab === 'compare' && (
        <CompareView runs={runs} allLogs={allLogs} />
      )}

      <TrainingPlanner />

      <div className="panel">
        <h2>Checkpoints</h2>
        <table>
          <thead>
            <tr><th>File</th><th>Size</th><th>Modified</th></tr>
          </thead>
          <tbody>
            {checkpoints.map(cp => (
              <tr key={cp.name}>
                <td className="mono">{cp.name}</td>
                <td>{cp.size_mb} MB</td>
                <td>{formatTime(cp.modified)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default App
