"""
SQLite storage layer for AgroNex.
Replaces the flat audit_log.csv with structured, queryable storage.
Generates readable IDs (BATCH-0001, ON-00001) instead of hashes.
"""

import sqlite3
from datetime import datetime

DB_PATH = "agronex.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            supplier_name TEXT,
            procurement_centre TEXT,
            date TEXT,
            onion_variety TEXT,
            quantity_received REAL,
            unit TEXT,
            intended_use TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY,
            batch_id TEXT,
            timestamp TEXT,
            image_path TEXT,
            ai_prediction TEXT,
            ai_confidence REAL,
            grade TEXT,
            human_decision TEXT,
            human_agreed INTEGER,
            correction_reason TEXT,
            defect_tags TEXT,
            FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
        )
    """)

    conn.commit()
    conn.close()


def _next_id(prefix, table, id_column):
    """Generates the next readable ID, e.g. BATCH-0001, ON-00001."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(f"SELECT {id_column} FROM {table} ORDER BY rowid DESC LIMIT 1")
    row = c.fetchone()
    conn.close()

    if row is None:
        next_num = 1
    else:
        last_id = row[0]
        last_num = int(last_id.split("-")[1])
        next_num = last_num + 1

    width = 4 if prefix == "BATCH" else 5
    return f"{prefix}-{str(next_num).zfill(width)}"


def create_batch(supplier_name, procurement_centre, onion_variety,
                  quantity_received, unit, intended_use):
    batch_id = _next_id("BATCH", "batches", "batch_id")
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO batches
        (batch_id, supplier_name, procurement_centre, date, onion_variety,
         quantity_received, unit, intended_use, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        batch_id, supplier_name, procurement_centre,
        datetime.now().strftime("%Y-%m-%d"), onion_variety,
        quantity_received, unit, intended_use, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    return batch_id


def create_sample(batch_id, image_path, ai_prediction, ai_confidence, grade):
    sample_id = _next_id("ON", "samples", "sample_id")
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO samples
        (sample_id, batch_id, timestamp, image_path, ai_prediction,
         ai_confidence, grade, human_decision, human_agreed,
         correction_reason, defect_tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
    """, (
        sample_id, batch_id, datetime.now().isoformat(), image_path,
        ai_prediction, ai_confidence, grade
    ))
    conn.commit()
    conn.close()
    return sample_id


def record_human_decision(sample_id, human_decision, correction_reason=None, defect_tags=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT ai_prediction FROM samples WHERE sample_id = ?", (sample_id,))
    row = c.fetchone()
    ai_prediction = row["ai_prediction"] if row else None
    human_agreed = 1 if human_decision == ai_prediction else 0

    c.execute("""
        UPDATE samples
        SET human_decision = ?, human_agreed = ?, correction_reason = ?, defect_tags = ?
        WHERE sample_id = ?
    """, (human_decision, human_agreed, correction_reason, defect_tags, sample_id))
    conn.commit()
    conn.close()


def get_batch_summary(batch_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM samples WHERE batch_id = ?", (batch_id,))
    rows = c.fetchall()
    conn.close()

    total = len(rows)
    healthy = sum(1 for r in rows if (r["human_decision"] or r["ai_prediction"]) == "healthy")
    defective = total - healthy
    corrected = sum(1 for r in rows if r["human_agreed"] == 0)

    return {
        "batch_id": batch_id,
        "total_samples": total,
        "healthy": healthy,
        "defective": defective,
        "human_corrections": corrected,
        "healthy_pct": round(healthy / total * 100, 1) if total else 0,
        "defective_pct": round(defective / total * 100, 1) if total else 0,
    }


def get_all_batches():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM batches ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]