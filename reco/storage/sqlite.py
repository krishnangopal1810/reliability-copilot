"""SQLite storage implementation."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from ..core.models import EvalRun, FailureCluster, Severity
from .base import ClusterMatch


class SQLiteStorage:
    """SQLite-based storage for runs and clusters."""
    
    def __init__(self, db_path: Path | str | None = None):
        """Initialize SQLite storage.
        
        Args:
            db_path: Path to database file. Defaults to .reco/data.db in project dir
        """
        if db_path is None:
            db_path = Path(".reco") / "data.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
    
    def _create_schema(self) -> None:
        """Create database tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            );
            
            CREATE TABLE IF NOT EXISTS clusters (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                label TEXT NOT NULL,
                description TEXT,
                severity TEXT NOT NULL,
                response_ids TEXT,
                embedding BLOB,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_clusters_run_id ON clusters(run_id);
            CREATE INDEX IF NOT EXISTS idx_clusters_label ON clusters(label);
        """)
        self.conn.commit()
    
    def save_run(self, run: EvalRun) -> None:
        """Persist an eval run."""
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (id, name, created_at, metadata) VALUES (?, ?, ?, ?)",
            (
                str(run.id),
                run.name,
                run.created_at.isoformat(),
                json.dumps(run.metadata),
            )
        )
        self.conn.commit()
    
    def save_clusters(
        self, 
        clusters: list[FailureCluster], 
        run_id: UUID,
        embeddings: list[NDArray[np.float32]]
    ) -> None:
        """Save clusters with their embeddings."""
        now = datetime.now(timezone.utc).isoformat()
        
        for cluster, embedding in zip(clusters, embeddings):
            self.conn.execute(
                """INSERT OR REPLACE INTO clusters 
                   (id, run_id, label, description, severity, response_ids, embedding, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(cluster.id),
                    str(run_id),
                    cluster.label,
                    cluster.description,
                    cluster.severity.value,
                    json.dumps(cluster.response_ids),
                    embedding.tobytes(),
                    now,
                )
            )
        self.conn.commit()
    
    def find_similar_clusters(
        self, 
        embedding: NDArray[np.float32], 
        threshold: float = 0.85,
        limit: int = 10
    ) -> list[ClusterMatch]:
        """Find historically similar clusters by cosine similarity."""
        cursor = self.conn.execute(
            """SELECT c.id, c.label, c.severity, c.embedding, c.run_id, r.created_at
               FROM clusters c
               JOIN runs r ON c.run_id = r.id
               WHERE c.embedding IS NOT NULL
               ORDER BY r.created_at DESC
               LIMIT 1000"""  # Cap to avoid scanning entire history
        )
        
        matches: list[ClusterMatch] = []
        label_counts: dict[str, int] = {}
        
        for row in cursor:
            # Deserialize embedding
            stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(embedding, stored_embedding)
            
            if similarity >= threshold:
                label = row["label"]
                label_counts[label] = label_counts.get(label, 0) + 1
                
                # Parse severity
                severity = Severity(row["severity"]) if row["severity"] else Severity.MEDIUM
                
                matches.append(ClusterMatch(
                    cluster_id=UUID(row["id"]),
                    label=label,
                    similarity=similarity,
                    run_id=UUID(row["run_id"]),
                    run_date=datetime.fromisoformat(row["created_at"]),
                    occurrence_count=label_counts[label],
                    severity=severity,
                ))
        
        # Sort by similarity descending, take top matches
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:limit]
    
    def get_cluster_history(self, label: str, limit: int = 10) -> list[ClusterMatch]:
        """Get history of clusters with the same label."""
        cursor = self.conn.execute(
            """SELECT c.id, c.label, c.severity, c.run_id, r.created_at
               FROM clusters c
               JOIN runs r ON c.run_id = r.id
               WHERE c.label = ?
               ORDER BY r.created_at DESC
               LIMIT ?""",
            (label, limit)
        )
        
        matches = []
        for i, row in enumerate(cursor, 1):
            severity = Severity(row["severity"]) if row["severity"] else Severity.MEDIUM
            matches.append(ClusterMatch(
                cluster_id=UUID(row["id"]),
                label=row["label"],
                similarity=1.0,  # Exact label match
                run_id=UUID(row["run_id"]),
                run_date=datetime.fromisoformat(row["created_at"]),
                occurrence_count=i,
                severity=severity,
            ))
        
        return matches
    
    def count_recent_occurrences(self, label: str, last_n_runs: int = 5) -> tuple[int, Optional[datetime]]:
        """Count occurrences of a label in recent runs.
        
        Returns:
            Tuple of (count, first_seen_date)
        """
        cursor = self.conn.execute(
            """SELECT COUNT(DISTINCT c.run_id) as count, MIN(r.created_at) as first_seen
               FROM clusters c
               JOIN runs r ON c.run_id = r.id
               WHERE c.label = ?
               AND c.run_id IN (
                   SELECT id FROM runs ORDER BY created_at DESC LIMIT ?
               )""",
            (label, last_n_runs)
        )
        
        row = cursor.fetchone()
        first_seen = None
        if row["first_seen"]:
            first_seen = datetime.fromisoformat(row["first_seen"])
        
        return row["count"] or 0, first_seen
    
    @staticmethod
    def _cosine_similarity(a: NDArray, b: NDArray) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))
    
    # Phase 2: Profile methods
    def get_failure_mode_stats(
        self, 
        limit_runs: int = 10
    ) -> list[tuple[str, int, int]]:
        """Get failure mode stats aggregated from recent runs.
        
        Returns:
            List of (label, occurrence_count, total_failures) tuples
        """
        # Get total failure count from recent runs
        cursor = self.conn.execute(
            """SELECT COUNT(*) as total 
               FROM clusters c
               WHERE c.run_id IN (
                   SELECT id FROM runs ORDER BY created_at DESC LIMIT ?
               )
               AND c.label != 'Uncategorized'
               AND c.label != 'Single Failure'""",
            (limit_runs,)
        )
        total_row = cursor.fetchone()
        total_clusters = total_row["total"] if total_row else 0
        
        # Get label counts
        cursor = self.conn.execute(
            """SELECT c.label, COUNT(*) as count
               FROM clusters c
               WHERE c.run_id IN (
                   SELECT id FROM runs ORDER BY created_at DESC LIMIT ?
               )
               AND c.label != 'Uncategorized'
               AND c.label != 'Single Failure'
               GROUP BY c.label
               ORDER BY count DESC""",
            (limit_runs,)
        )
        
        return [(row["label"], row["count"], total_clusters) for row in cursor]
    
    def get_pass_rate_history(
        self, 
        limit: int = 10
    ) -> list[tuple[datetime, float, int]]:
        """Get pass rate history from recent runs.
        
        Note: We don't store individual responses, so we estimate from cluster data.
        For now, returns runs with placeholder pass rates.
        """
        cursor = self.conn.execute(
            """SELECT r.created_at, r.name, COUNT(c.id) as failure_count
               FROM runs r
               LEFT JOIN clusters c ON c.run_id = r.id
               GROUP BY r.id
               ORDER BY r.created_at DESC
               LIMIT ?""",
            (limit,)
        )
        
        # Return in chronological order (oldest first)
        results = []
        for row in cursor:
            created_at = datetime.fromisoformat(row["created_at"])
            # Placeholder: we'd need to store response counts for real pass rate
            failure_count = row["failure_count"] or 0
            results.append((created_at, failure_count, 0))
        
        return list(reversed(results))
    
    def get_run_count(self) -> int:
        """Get total number of runs in storage."""
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM runs")
        row = cursor.fetchone()
        return row["count"] if row else 0
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
