"""
Encrypted local storage using SQLCipher.
Stores patient records and query history securely.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from pysqlcipher3 import dbapi2 as sqlite
except ImportError:
    # Fallback to sqlcipher3_binary if pysqlcipher3 not available
    import sqlcipher3 as sqlite


class EncryptedStorage:
    """
    Encrypted SQLite storage using SQLCipher.
    All patient data is encrypted at rest.
    """

    def __init__(self, db_path: str = "data/local.db", key: str | None = None):
        """
        Initialize encrypted storage.

        Args:
            db_path: Path to database file
            key: Encryption key (will use env var if not provided)
        """
        self.db_path = db_path
        self.key = key or os.getenv("STORAGE_KEY", "default-dev-key-change-in-prod")

        # Create data directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite.Connection:
        """Get database connection with encryption enabled"""
        conn = sqlite.connect(self.db_path)
        # Sanitize key by escaping single quotes (SQLCipher requirement)
        # The key should be from a trusted source (environment variable or config)
        safe_key = self.key.replace("'", "''")
        conn.execute(f"PRAGMA key = '{safe_key}'")
        conn.execute("PRAGMA cipher_compatibility = 4")
        return conn

    def _init_db(self):
        """Initialize database schema"""
        conn = self._get_connection()
        try:
            # Patient records table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patient_records (
                    patient_id TEXT PRIMARY KEY,
                    age INTEGER NOT NULL,
                    sex TEXT NOT NULL,
                    name TEXT,
                    date_of_birth TEXT,
                    conditions TEXT,
                    medications TEXT,
                    allergies TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Query history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    patient_id TEXT,
                    symptoms TEXT NOT NULL,
                    vitals TEXT,
                    urgency_level TEXT,
                    is_emergency INTEGER DEFAULT 0,
                    predictions TEXT,
                    fingerprint TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patient_records (patient_id)
                )
            """)

            # Lab results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lab_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reference_range TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patient_records (patient_id)
                )
            """)

            # Patient history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patient_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    symptoms TEXT NOT NULL,
                    diagnosis TEXT,
                    notes TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patient_records (patient_id)
                )
            """)

            conn.commit()
        finally:
            conn.close()

    def store_patient(self, patient_data: dict[str, Any]) -> bool:
        """
        Store or update patient record.

        Args:
            patient_data: Patient record dictionary

        Returns:
            True if successful
        """
        conn = self._get_connection()
        try:
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT OR REPLACE INTO patient_records
                (patient_id, age, sex, name, date_of_birth, conditions, 
                 medications, allergies, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                        COALESCE((SELECT created_at FROM patient_records WHERE patient_id = ?), ?), ?)
            """,
                (
                    patient_data["patient_id"],
                    patient_data["age"],
                    patient_data["sex"],
                    patient_data.get("name"),
                    patient_data.get("date_of_birth"),
                    json.dumps(patient_data.get("conditions", [])),
                    json.dumps(patient_data.get("medications", [])),
                    json.dumps(patient_data.get("allergies", [])),
                    patient_data["patient_id"],
                    now,
                    now,
                ),
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing patient: {e}")
            return False
        finally:
            conn.close()

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        """
        Retrieve patient record.

        Args:
            patient_id: Patient identifier

        Returns:
            Patient record dictionary or None
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT patient_id, age, sex, name, date_of_birth, 
                       conditions, medications, allergies, created_at, updated_at
                FROM patient_records WHERE patient_id = ?
            """,
                (patient_id,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "patient_id": row[0],
                    "age": row[1],
                    "sex": row[2],
                    "name": row[3],
                    "date_of_birth": row[4],
                    "conditions": json.loads(row[5]) if row[5] else [],
                    "medications": json.loads(row[6]) if row[6] else [],
                    "allergies": json.loads(row[7]) if row[7] else [],
                    "created_at": row[8],
                    "updated_at": row[9],
                }
            return None
        finally:
            conn.close()

    def store_query(self, query_data: dict[str, Any]) -> int | None:
        """
        Store query history entry.

        Args:
            query_data: Query information

        Returns:
            Query ID if successful, None otherwise
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO query_history
                (device_id, patient_id, symptoms, vitals, urgency_level, 
                 is_emergency, predictions, fingerprint, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    query_data["device_id"],
                    query_data.get("patient_id"),
                    json.dumps(query_data["symptoms"]),
                    json.dumps(query_data.get("vitals")),
                    query_data.get("urgency_level"),
                    1 if query_data.get("is_emergency") else 0,
                    json.dumps(query_data.get("predictions")),
                    query_data.get("fingerprint"),
                    query_data.get("timestamp", datetime.now(UTC).isoformat()),
                ),
            )
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error storing query: {e}")
            return None
        finally:
            conn.close()

    def get_query_history(
        self, device_id: str | None = None, patient_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Retrieve query history.

        Args:
            device_id: Filter by device ID
            patient_id: Filter by patient ID
            limit: Maximum number of records

        Returns:
            List of query records
        """
        conn = self._get_connection()
        try:
            where_clauses = []
            params = []

            if device_id:
                where_clauses.append("device_id = ?")
                params.append(device_id)
            if patient_id:
                where_clauses.append("patient_id = ?")
                params.append(patient_id)

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            cursor = conn.execute(
                f"""
                SELECT id, device_id, patient_id, symptoms, vitals, urgency_level,
                       is_emergency, predictions, fingerprint, timestamp
                FROM query_history
                {where_sql}
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (*params, limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row[0],
                        "device_id": row[1],
                        "patient_id": row[2],
                        "symptoms": json.loads(row[3]) if row[3] else [],
                        "vitals": json.loads(row[4]) if row[4] else None,
                        "urgency_level": row[5],
                        "is_emergency": bool(row[6]),
                        "predictions": json.loads(row[7]) if row[7] else None,
                        "fingerprint": row[8],
                        "timestamp": row[9],
                    }
                )
            return results
        finally:
            conn.close()

    def store_lab_result(self, lab_data: dict[str, Any]) -> bool:
        """
        Store lab result.

        Args:
            lab_data: Lab result data

        Returns:
            True if successful
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO lab_results
                (patient_id, test_name, value, unit, timestamp, reference_range)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    lab_data["patient_id"],
                    lab_data["test_name"],
                    lab_data["value"],
                    lab_data["unit"],
                    lab_data.get("timestamp", datetime.now(UTC).isoformat()),
                    lab_data.get("reference_range"),
                ),
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing lab result: {e}")
            return False
        finally:
            conn.close()

    def get_lab_results(self, patient_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Retrieve lab results for patient.

        Args:
            patient_id: Patient identifier
            limit: Maximum number of results

        Returns:
            List of lab results
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, test_name, value, unit, timestamp, reference_range
                FROM lab_results
                WHERE patient_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (patient_id, limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row[0],
                        "test_name": row[1],
                        "value": row[2],
                        "unit": row[3],
                        "timestamp": row[4],
                        "reference_range": row[5],
                    }
                )
            return results
        finally:
            conn.close()

    def clear_all_data(self):
        """Clear all data from database (for testing only)"""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM lab_results")
            conn.execute("DELETE FROM patient_history")
            conn.execute("DELETE FROM query_history")
            conn.execute("DELETE FROM patient_records")
            conn.commit()
        finally:
            conn.close()
