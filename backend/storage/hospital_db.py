"""
Hospital DB — Local SQLite-based medical data store.

Simulates a hospital database for federated learning.
Stores anonymised medical records (from user interactions and reports)
that can be used for federated model training.

All data stored locally. This is the "edge storage" component
referenced in the system architecture.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

BACKEND_ROOT = Path(__file__).parent.parent  # backend/
DATA_DIR = BACKEND_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "hospital_local.db"


def _get_conn() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS medical_records (
            id TEXT PRIMARY KEY,
            record_type TEXT NOT NULL,        -- 'consultation' | 'report' | 'lab_result'
            category TEXT,                     -- predicted category from classifier
            severity TEXT,                     -- 'critical' | 'moderate' | 'low'
            symptoms_hash TEXT,                -- SHA-256 hash (not raw symptoms, for privacy)
            council_summary TEXT,              -- anonymised council output
            confidence REAL,
            timestamp TEXT NOT NULL,
            metadata TEXT                      -- JSON string for extra fields
        );

        CREATE TABLE IF NOT EXISTS report_embeddings (
            id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding_vector BLOB,             -- numpy array as bytes
            timestamp TEXT NOT NULL,
            FOREIGN KEY (report_id) REFERENCES medical_records(id)
        );

        CREATE TABLE IF NOT EXISTS federated_contributions (
            id TEXT PRIMARY KEY,
            record_id TEXT,
            gradient_hash TEXT,
            dp_noise_level REAL,
            contributed_at TEXT NOT NULL,
            aggregation_round INTEGER,
            status TEXT DEFAULT 'pending',      -- 'pending' | 'aggregated' | 'rejected'
            FOREIGN KEY (record_id) REFERENCES medical_records(id)
        );

        CREATE INDEX IF NOT EXISTS idx_records_category ON medical_records(category);
        CREATE INDEX IF NOT EXISTS idx_records_timestamp ON medical_records(timestamp);
        CREATE INDEX IF NOT EXISTS idx_contributions_status ON federated_contributions(status);
    """)
    conn.commit()
    conn.close()
    print("[HospitalDB] Database initialized.")


def store_consultation(
    category: str,
    severity: str,
    symptoms_hash: str,
    council_summary: str,
    confidence: float,
    metadata: Optional[dict] = None,
) -> str:
    """Store a completed consultation record."""
    record_id = f"cons_{uuid.uuid4().hex[:8]}"
    conn = _get_conn()
    conn.execute(
        """INSERT INTO medical_records
           (id, record_type, category, severity, symptoms_hash,
            council_summary, confidence, timestamp, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record_id, "consultation", category, severity,
            symptoms_hash, council_summary, confidence,
            datetime.now().isoformat(),
            json.dumps(metadata or {}),
        ),
    )
    conn.commit()
    conn.close()
    return record_id


def store_report_record(
    report_id: str,
    category: str,
    summary: str,
    metadata: Optional[dict] = None,
) -> str:
    """Store a medical report record."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO medical_records
           (id, record_type, category, severity, symptoms_hash,
            council_summary, confidence, timestamp, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            report_id, "report", category, "n/a",
            "", summary, 0.0,
            datetime.now().isoformat(),
            json.dumps(metadata or {}),
        ),
    )
    conn.commit()
    conn.close()
    return report_id


def log_federated_contribution(
    record_id: str,
    gradient_hash: str,
    dp_noise_level: float,
    aggregation_round: int = 0,
) -> str:
    """Log a federated learning contribution."""
    contrib_id = f"fed_{uuid.uuid4().hex[:8]}"
    conn = _get_conn()
    conn.execute(
        """INSERT INTO federated_contributions
           (id, record_id, gradient_hash, dp_noise_level,
            contributed_at, aggregation_round, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            contrib_id, record_id, gradient_hash,
            dp_noise_level, datetime.now().isoformat(),
            aggregation_round, "pending",
        ),
    )
    conn.commit()
    conn.close()
    return contrib_id


def get_records(record_type: Optional[str] = None, limit: int = 50) -> list[dict]:
    """Get medical records, optionally filtered by type."""
    conn = _get_conn()
    if record_type:
        rows = conn.execute(
            "SELECT * FROM medical_records WHERE record_type = ? ORDER BY timestamp DESC LIMIT ?",
            (record_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM medical_records ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_db_stats() -> dict:
    """Get database statistics."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM medical_records").fetchone()[0]
    consultations = conn.execute(
        "SELECT COUNT(*) FROM medical_records WHERE record_type='consultation'"
    ).fetchone()[0]
    reports = conn.execute(
        "SELECT COUNT(*) FROM medical_records WHERE record_type='report'"
    ).fetchone()[0]
    contributions = conn.execute(
        "SELECT COUNT(*) FROM federated_contributions"
    ).fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM federated_contributions WHERE status='pending'"
    ).fetchone()[0]
    conn.close()

    return {
        "total_records": total,
        "consultations": consultations,
        "reports": reports,
        "federated_contributions": contributions,
        "pending_aggregations": pending,
    }


# Initialize on import
init_db()
