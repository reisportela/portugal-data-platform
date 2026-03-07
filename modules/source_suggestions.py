import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


REVIEW_STATUSES = ["pending_review", "approved", "rejected"]


class SourceSuggestionsStore:
    """
    Moderated inbox for user-submitted data sources.

    Suggestions are intentionally stored outside the public dataset catalog.
    They must be reviewed and manually promoted before appearing in search.
    """

    def __init__(self, db_path: str = "data/source_suggestions.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    institution TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    link TEXT NOT NULL,
                    description TEXT NOT NULL,
                    keywords TEXT DEFAULT '',
                    source_type TEXT DEFAULT '',
                    access TEXT DEFAULT '',
                    spatial_level TEXT DEFAULT '',
                    submitter_name TEXT DEFAULT '',
                    submitter_email TEXT DEFAULT '',
                    submission_notes TEXT DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT 'pending_review',
                    curator_notes TEXT DEFAULT '',
                    submitted_at TEXT NOT NULL,
                    reviewed_at TEXT DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_source_suggestions_review_status
                ON source_suggestions(review_status, submitted_at)
                """
            )

    @staticmethod
    def _clean(value: object) -> str:
        return str(value).strip()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def submit_suggestion(self, suggestion: Dict[str, object]) -> int:
        record = {
            "dataset_name": self._clean(suggestion.get("dataset_name", "")),
            "institution": self._clean(suggestion.get("institution", "")),
            "domain": self._clean(suggestion.get("domain", "")),
            "link": self._clean(suggestion.get("link", "")),
            "description": self._clean(suggestion.get("description", "")),
            "keywords": self._clean(suggestion.get("keywords", "")),
            "source_type": self._clean(suggestion.get("source_type", "")),
            "access": self._clean(suggestion.get("access", "")),
            "spatial_level": self._clean(suggestion.get("spatial_level", "")),
            "submitter_name": self._clean(suggestion.get("submitter_name", "")),
            "submitter_email": self._clean(suggestion.get("submitter_email", "")),
            "submission_notes": self._clean(suggestion.get("submission_notes", "")),
            "review_status": "pending_review",
            "curator_notes": "",
            "submitted_at": self._timestamp(),
            "reviewed_at": "",
        }

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO source_suggestions (
                    dataset_name,
                    institution,
                    domain,
                    link,
                    description,
                    keywords,
                    source_type,
                    access,
                    spatial_level,
                    submitter_name,
                    submitter_email,
                    submission_notes,
                    review_status,
                    curator_notes,
                    submitted_at,
                    reviewed_at
                )
                VALUES (
                    :dataset_name,
                    :institution,
                    :domain,
                    :link,
                    :description,
                    :keywords,
                    :source_type,
                    :access,
                    :spatial_level,
                    :submitter_name,
                    :submitter_email,
                    :submission_notes,
                    :review_status,
                    :curator_notes,
                    :submitted_at,
                    :reviewed_at
                )
                """,
                record,
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_suggestions(self, review_status: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM source_suggestions"
        params = []
        if review_status:
            query += " WHERE review_status = ?"
            params.append(review_status)
        query += " ORDER BY submitted_at DESC, id DESC"

        with self._connect() as conn:
            frame = pd.read_sql_query(query, conn, params=params)
        return frame.fillna("")

    def pending_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM source_suggestions WHERE review_status = 'pending_review'"
            ).fetchone()
        return int(row["total"]) if row else 0

    def update_review(self, suggestion_id: int, review_status: str, curator_notes: str = "") -> None:
        if review_status not in REVIEW_STATUSES:
            raise ValueError(f"Invalid review status: {review_status}")

        reviewed_at = self._timestamp() if review_status != "pending_review" else ""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE source_suggestions
                SET review_status = ?,
                    curator_notes = ?,
                    reviewed_at = ?
                WHERE id = ?
                """,
                (self._clean(review_status), self._clean(curator_notes), reviewed_at, int(suggestion_id)),
            )
            conn.commit()

    def approved_catalog_candidates(self) -> pd.DataFrame:
        approved = self.list_suggestions("approved")
        if approved.empty:
            return pd.DataFrame(
                columns=[
                    "dataset_name",
                    "institution",
                    "domain",
                    "link",
                    "description",
                    "keywords",
                    "source_type",
                    "access",
                    "spatial_level",
                ]
            )

        columns = [
            "dataset_name",
            "institution",
            "domain",
            "link",
            "description",
            "keywords",
            "source_type",
            "access",
            "spatial_level",
        ]
        return approved[columns].copy().reset_index(drop=True)
