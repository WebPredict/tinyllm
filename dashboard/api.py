"""
Simple API server that reads from the SQLite training database
and serves data to the React dashboard.
"""

import sqlite3
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path(__file__).parent.parent / "logs" / "training.db"


def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    return db


@app.get("/api/runs")
def get_runs():
    """List all training runs."""
    db = get_db()
    rows = db.execute("""
        SELECT run_id, config, started, finished, status,
               final_train_loss, final_val_loss
        FROM runs ORDER BY started DESC
    """).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/runs/{run_id}/logs")
def get_run_logs(run_id: str):
    """Get training logs for a specific run."""
    db = get_db()
    rows = db.execute("""
        SELECT step, timestamp, train_loss, val_loss,
               learning_rate, tokens_seen, tokens_per_second, sample_output
        FROM training_logs
        WHERE run_id = ?
        ORDER BY step
    """, (run_id,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/runs/{run_id}/config")
def get_run_config(run_id: str):
    """Get config for a specific run."""
    db = get_db()
    row = db.execute(
        "SELECT config FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row:
        return json.loads(row["config"])
    return {}


@app.get("/api/evals")
def get_evals():
    """Get all eval results."""
    db = get_db()
    try:
        rows = db.execute("""
            SELECT run_id, checkpoint, timestamp, model_params,
                   eval_type, metric_name, metric_value
            FROM eval_results
            ORDER BY timestamp DESC
        """).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


@app.get("/api/evals/{run_id}")
def get_run_evals(run_id: str):
    """Get eval results for a specific run."""
    db = get_db()
    try:
        rows = db.execute("""
            SELECT eval_type, metric_name, metric_value
            FROM eval_results
            WHERE run_id = ?
        """, (run_id,)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


@app.get("/api/checkpoints")
def get_checkpoints():
    """List all checkpoint files."""
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    files = sorted(checkpoint_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [
        {
            "name": f.name,
            "size_mb": round(f.stat().st_size / 1_000_000, 1),
            "modified": f.stat().st_mtime,
        }
        for f in files
    ]
