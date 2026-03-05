import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
from rapidfuzz import fuzz


@dataclass
class SearchResult:
    row: Dict
    score: float
    matched_fields: List[str]


class SearchEngine:
    """
    Full working search engine for Portugal Data Platform.

    Expected CSV columns:
    - dataset_name
    - institution
    - domain
    - link

    Optional columns:
    - unit
    - spatial_level
    - time_coverage
    - access
    """

    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path).fillna("")
        self.required_columns = ["dataset_name", "institution", "domain", "link"]
        missing = [c for c in self.required_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.search_columns = [
            c for c in [
                "dataset_name",
                "institution",
                "domain",
                "unit",
                "spatial_level",
                "time_coverage",
                "access",
            ] if c in self.df.columns
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = SearchEngine._normalize(text)
        return [t for t in re.split(r"[^a-z0-9áàâãéèêíïóôõöúç-]+", text) if t]

    def _score_row(self, query: str, row: pd.Series) -> Tuple[float, List[str]]:
        q = self._normalize(query)
        q_tokens = self._tokenize(query)

        total_score = 0.0
        matched_fields = []

        # Weights by field importance
        weights = {
            "dataset_name": 5.0,
            "domain": 3.0,
            "institution": 2.5,
            "unit": 1.5,
            "spatial_level": 1.5,
            "time_coverage": 0.5,
            "access": 1.0,
        }

        for col in self.search_columns:
            text = self._normalize(row[col])
            if not text:
                continue

            field_score = 0.0

            # Exact substring match
            if q and q in text:
                field_score += 100

            # Token coverage
            token_hits = sum(1 for t in q_tokens if t in text)
            if q_tokens:
                field_score += 40 * (token_hits / len(q_tokens))

            # Fuzzy partial ratio
            fuzzy = fuzz.partial_ratio(q, text)
            field_score += 0.6 * fuzzy

            # Prefix bonus on dataset_name
            if col == "dataset_name" and text.startswith(q):
                field_score += 30

            # If the field is actually relevant, keep track
            if field_score >= 60:
                matched_fields.append(col)

            total_score += field_score * weights.get(col, 1.0)

        return total_score, matched_fields

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 120.0,
        domain_filter: str = None,
        institution_filter: str = None,
    ) -> pd.DataFrame:
        """
        Returns a ranked DataFrame of matches.
        """
        query = self._normalize(query)
        if not query:
            return self.df.head(0).copy()

        working = self.df.copy()

        if domain_filter:
            dfilt = self._normalize(domain_filter)
            working = working[
                working["domain"].astype(str).str.lower().str.contains(dfilt, na=False)
            ]

        if institution_filter:
            ifilt = self._normalize(institution_filter)
            working = working[
                working["institution"].astype(str).str.lower().str.contains(ifilt, na=False)
            ]

        results = []
        for _, row in working.iterrows():
            score, matched_fields = self._score_row(query, row)
            if score >= min_score:
                item = row.to_dict()
                item["_score"] = round(score, 2)
                item["_matched_fields"] = ", ".join(matched_fields)
                results.append(item)

        if not results:
            # Fallback: return best fuzzy candidates even if below threshold
            fallback = []
            for _, row in working.iterrows():
                score, matched_fields = self._score_row(query, row)
                item = row.to_dict()
                item["_score"] = round(score, 2)
                item["_matched_fields"] = ", ".join(matched_fields)
                fallback.append(item)

            out = pd.DataFrame(fallback).sort_values("_score", ascending=False).head(limit)
            return out[out["_score"] > 0].reset_index(drop=True)

        out = pd.DataFrame(results).sort_values("_score", ascending=False).head(limit)
        return out.reset_index(drop=True)

    def suggest(self, query: str, limit: int = 8) -> List[str]:
        """
        Returns simple search suggestions based on dataset names, domains, and institutions.
        """
        query = self._normalize(query)
        if not query:
            base = list(self.df["dataset_name"].astype(str).head(limit))
            return base

        pool = set()
        for col in ["dataset_name", "domain", "institution"]:
            if col in self.df.columns:
                for value in self.df[col].astype(str).unique():
                    v = self._normalize(value)
                    if query in v or fuzz.partial_ratio(query, v) >= 70:
                        pool.add(value)

        suggestions = sorted(pool)[:limit]
        return suggestions

    @staticmethod
    def highlight_text(text: str, query: str) -> str:
        """
        Returns HTML with matched query fragments highlighted.
        Safe enough for st.markdown(..., unsafe_allow_html=True) in Streamlit.
        """
        if not text or not query:
            return str(text)

        out = str(text)
        for token in SearchEngine._tokenize(query):
            pattern = re.compile(re.escape(token), re.IGNORECASE)
            out = pattern.sub(
                lambda m: f"<mark>{m.group(0)}</mark>",
                out
            )
        return out
