"""
core/metrics.py
===============
Dual-sink EvalLogger: writes to SQLite (results/eval_ledger.db) and
streaming JSONL (results/eval_ledger.jsonl) simultaneously.

Every evaluation run is tagged with a UUIDv4 run_id, ISO-8601 timestamp,
and full config metadata for reproducibility.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
    run_id          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    git_hash        TEXT,
    model_family    TEXT,
    modality        TEXT,
    dataset         TEXT,
    seed            INTEGER,
    quant_mode      TEXT,
    metric_name     TEXT,
    metric_value    REAL,
    weight_bits     REAL,
    act_bits        REAL,
    eff_bits        REAL,
    peak_vram_mb    REAL,
    mean_kernel_ms  REAL,
    speedup         REAL,
    trigger_stats   TEXT,
    PRIMARY KEY (run_id, metric_name)
);
"""


def _get_git_hash() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class EvalLogger:
    """
    Thread-safe dual-sink logger.

    Usage:
        logger = EvalLogger(results_dir="results")
        logger.log(model_family="gpt2", quant_mode="mxfp4", metric_name="ppl",
                   metric_value=109.72, eff_bits=4.25, seed=42)
    """

    def __init__(self, results_dir: str = "results"):
        self._dir = Path(results_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._dir / "eval_ledger.db"
        self._jsonl_path = self._dir / "eval_ledger.jsonl"
        self._git_hash = _get_git_hash()

        # Initialise DB
        conn = self._connect()
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def log(
        self,
        *,
        model_family: str,
        modality: str = "language",
        dataset: str = "wikitext-2-raw-v1",
        seed: int = 42,
        quant_mode: str,
        metric_name: str,
        metric_value: float,
        weight_bits: float = 4.25,
        act_bits: float = 4.25,
        eff_bits: float = 4.25,
        peak_vram_mb: float = 0.0,
        mean_kernel_ms: float = 0.0,
        speedup: float = 1.0,
        trigger_stats: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Log one metric for one run. Returns the run_id."""
        run_id = run_id or str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        trigger_json = json.dumps(trigger_stats or {})

        row = dict(
            run_id=run_id,
            timestamp=ts,
            git_hash=self._git_hash,
            model_family=model_family,
            modality=modality,
            dataset=dataset,
            seed=seed,
            quant_mode=quant_mode,
            metric_name=metric_name,
            metric_value=metric_value,
            weight_bits=weight_bits,
            act_bits=act_bits,
            eff_bits=eff_bits,
            peak_vram_mb=peak_vram_mb,
            mean_kernel_ms=mean_kernel_ms,
            speedup=speedup,
            trigger_stats=trigger_json,
        )

        # SQLite
        conn = self._connect()
        placeholders = ", ".join(["?"] * len(row))
        cols = ", ".join(row.keys())
        conn.execute(
            f"INSERT OR REPLACE INTO evaluations ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
        conn.commit()
        conn.close()

        # JSONL
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        return run_id

    def query(self, sql: str) -> list:
        conn = self._connect()
        cur = conn.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows

    def print_table(self, sql: Optional[str] = None) -> None:
        sql = sql or (
            "SELECT model_family, quant_mode, metric_name, "
            "ROUND(metric_value,2), ROUND(eff_bits,2) "
            "FROM evaluations ORDER BY model_family, eff_bits"
        )
        rows = self.query(sql)
        if not rows:
            print("(no rows)")
            return
        for r in rows:
            print("\t".join(str(c) for c in r))
