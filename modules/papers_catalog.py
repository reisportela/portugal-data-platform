import html
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
from rapidfuzz import fuzz


STOPWORDS = {
    "a",
    "an",
    "and",
    "article",
    "as",
    "at",
    "by",
    "data",
    "dataset",
    "datasets",
    "de",
    "do",
    "evidence",
    "for",
    "from",
    "in",
    "of",
    "on",
    "paper",
    "portugal",
    "study",
    "the",
    "to",
    "using",
    "with",
}

FIELD_LABELS = {
    "paper_title": "title",
    "authors": "authors",
    "publication": "publication",
    "topic": "topic",
    "methods": "methods",
    "datasets_used": "datasets",
    "data_notes": "data notes",
    "aliases": "related terms",
}

FIELD_WEIGHTS = {
    "paper_title": {"phrase": 60.0, "token": 16.0, "prefix": 10.0, "fuzzy": 7.0},
    "authors": {"phrase": 30.0, "token": 8.0, "prefix": 4.0, "fuzzy": 3.0},
    "publication": {"phrase": 26.0, "token": 6.0, "prefix": 3.0, "fuzzy": 2.0},
    "topic": {"phrase": 38.0, "token": 11.0, "prefix": 6.0, "fuzzy": 4.0},
    "methods": {"phrase": 16.0, "token": 5.0, "prefix": 0.0, "fuzzy": 0.0},
    "datasets_used": {"phrase": 44.0, "token": 12.0, "prefix": 7.0, "fuzzy": 4.0},
    "data_notes": {"phrase": 18.0, "token": 6.0, "prefix": 0.0, "fuzzy": 0.0},
    "aliases": {"phrase": 30.0, "token": 9.0, "prefix": 5.0, "fuzzy": 3.0},
}

CONCEPT_GROUPS = {
    "labour": {
        "employment",
        "gender wage gap",
        "job mobility",
        "joblessness",
        "labour",
        "labour market",
        "training",
        "unemployment",
        "wage",
        "wages",
        "workers",
    },
    "income": {
        "deprivation",
        "income",
        "inequality",
        "living conditions",
        "old age poverty",
        "poverty",
    },
    "innovation": {
        "community innovation survey",
        "firm growth",
        "human capital",
        "innovation",
        "technology",
    },
    "finance": {
        "bank",
        "banks",
        "credit",
        "credit channel",
        "credit register",
        "default risk",
        "financial distress",
        "leverage",
        "loans",
        "monetary policy",
    },
    "health": {
        "health",
        "health care",
        "health outcomes",
        "migrants",
        "national health survey",
    },
    "macro": {
        "gdp",
        "human capital multipliers",
        "input output",
        "macroeconomics",
        "structural change",
    },
    "demography": {
        "census",
        "city size",
        "population",
        "regional",
        "urban",
    },
    "prices": {
        "consumer prices",
        "cpi",
        "inflation",
        "price setting",
        "prices",
    },
}

FEATURED_PAPERS = [
    "On the Returns to Training in Portugal",
    "Inter-industry Wage Premia in Portugal: Evidence from EU-SILC data",
    "Coworker Networks and the Labor Market Outcomes of Displaced Workers: Evidence from Portugal",
    "Innovation strategy by firms: do innovative firms grow more?",
    "Estimating individuals' default risk in Portugal",
    "The Structure and Evolution of Production, Employment and Human Capital in Portugal: an Input-Output Approach",
]


@dataclass
class PaperSearchResult:
    row: Dict[str, object]
    score: float
    matched_fields: List[str]
    match_reason: str
    confidence: str


class PapersCatalog:
    """
    Curated paper layer for the Portugal Data Platform.

    Required columns:
    - paper_title
    - authors
    - year
    - publication
    - publication_type
    - topic
    - methods
    - dataset_keys
    - datasets_used
    - data_notes
    - paper_link
    - access
    """

    def __init__(self, file_path: str):
        catalog_path = Path(file_path)
        self.df = pd.read_csv(catalog_path).fillna("")
        self.required_columns = [
            "paper_title",
            "authors",
            "year",
            "publication",
            "publication_type",
            "topic",
            "methods",
            "dataset_keys",
            "datasets_used",
            "data_notes",
            "paper_link",
            "access",
        ]
        missing = [column for column in self.required_columns if column not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df["year"] = pd.to_numeric(self.df["year"], errors="raise").astype(int)
        self._enrich_catalog()
        self.search_columns = [
            column
            for column in [
                "paper_title",
                "authors",
                "publication",
                "topic",
                "methods",
                "datasets_used",
                "data_notes",
                "aliases",
            ]
            if column in self.df.columns
        ]
        self.public_columns = [
            column
            for column in [
                "paper_title",
                "authors",
                "year",
                "publication",
                "publication_type",
                "topic",
                "methods",
                "datasets_used",
                "data_notes",
                "access",
                "paper_link",
            ]
            if column in self.df.columns
        ]
        self.dataset_options = self._sorted_terms(self.df["dataset_keys"])
        self.topic_options = self._sorted_terms(self.df["topic"])
        self.publication_types = sorted(self.df["publication_type"].astype(str).unique())
        self.year_min = int(self.df["year"].min())
        self.year_max = int(self.df["year"].max())
        self._indexed_rows = [self._index_row(row) for _, row in self.df.iterrows()]
        self._idf = self._build_idf()

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKD", str(text))
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = text.lower().replace("&", " and ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        normalized = PapersCatalog._normalize(text)
        return [token for token in normalized.split(" ") if token]

    @staticmethod
    def _split_terms(value: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(value, str):
            return [part.strip() for part in re.split(r"[|;]", value) if part.strip()]
        return [str(part).strip() for part in value if str(part).strip()]

    @staticmethod
    def _merge_terms(*groups: Iterable[str]) -> List[str]:
        merged: List[str] = []
        seen: Set[str] = set()
        for group in groups:
            for term in group:
                normalized = PapersCatalog._normalize(term)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(str(term).strip())
        return merged

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 85:
            return "High"
        if score >= 45:
            return "Good"
        return "Broad"

    @staticmethod
    def _sorted_terms(series: pd.Series) -> List[str]:
        terms = set()
        for value in series.astype(str):
            terms.update(PapersCatalog._split_terms(value))
        return sorted(terms)

    def _extract_concepts(self, text: str) -> Set[str]:
        normalized = self._normalize(text)
        if not normalized:
            return set()

        tokens = set(self._tokenize(normalized))
        concepts = set()
        for concept, synonyms in CONCEPT_GROUPS.items():
            for synonym in synonyms:
                synonym_normalized = self._normalize(synonym)
                synonym_tokens = set(self._tokenize(synonym_normalized))
                if not synonym_tokens:
                    continue
                if len(synonym_tokens) > 1:
                    if synonym_normalized in normalized:
                        concepts.add(concept)
                        break
                elif synonym_normalized in tokens:
                    concepts.add(concept)
                    break
        return concepts

    def _enrich_catalog(self) -> None:
        aliases_list = []
        concepts_list = []

        for _, row in self.df.iterrows():
            aliases = self._merge_terms(
                self._split_terms(row["topic"]),
                self._split_terms(row["methods"]),
                self._split_terms(row["dataset_keys"]),
                self._split_terms(row["datasets_used"]),
                self._split_terms(row["authors"]),
                [row["publication"], row["publication_type"]],
            )
            concepts = self._merge_terms(
                self._extract_concepts(
                    " ".join(
                        [
                            str(row["paper_title"]),
                            str(row["topic"]),
                            str(row["methods"]),
                            str(row["dataset_keys"]),
                            str(row["datasets_used"]),
                            str(row["data_notes"]),
                        ]
                    )
                )
            )
            aliases_list.append(" | ".join(aliases))
            concepts_list.append(" | ".join(concepts))

        self.df["aliases"] = aliases_list
        self.df["concepts"] = concepts_list

    def _index_row(self, row: pd.Series) -> Dict[str, object]:
        item = row.to_dict()
        field_text = {}
        field_tokens = {}
        all_tokens: Set[str] = set()
        for column in self.search_columns:
            normalized = self._normalize(item.get(column, ""))
            tokens = set(self._tokenize(normalized))
            field_text[column] = normalized
            field_tokens[column] = tokens
            all_tokens.update(tokens)

        dataset_keys = self._split_terms(item.get("dataset_keys", ""))
        normalized_dataset_keys = [self._normalize(value) for value in dataset_keys if self._normalize(value)]
        concepts = set(self._split_terms(item.get("concepts", ""))) | self._extract_concepts(" ".join(field_text.values()))
        return {
            "row": item,
            "field_text": field_text,
            "field_tokens": field_tokens,
            "all_tokens": all_tokens,
            "concepts": concepts,
            "dataset_keys": dataset_keys,
            "normalized_dataset_keys": normalized_dataset_keys,
        }

    def _build_idf(self) -> Dict[str, float]:
        counts: Counter = Counter()
        for indexed_row in self._indexed_rows:
            counts.update(indexed_row["all_tokens"] | indexed_row["concepts"])

        n_rows = max(len(self._indexed_rows), 1)
        return {
            token: math.log((1 + n_rows) / (1 + df)) + 1.0
            for token, df in counts.items()
        }

    def _informative_query_tokens(self, query: str) -> List[str]:
        raw_tokens = self._tokenize(query)
        filtered = [token for token in raw_tokens if token not in STOPWORDS]
        return filtered or raw_tokens

    def _best_fuzzy_hits(
        self,
        query_tokens: Sequence[str],
        field_tokens: Set[str],
        include_broad_matches: bool,
    ) -> List[Tuple[str, str, float]]:
        if not include_broad_matches:
            return []

        hits = []
        for query_token in query_tokens:
            if len(query_token) < 5 or query_token in field_tokens:
                continue
            best_score = 0.0
            best_token = ""
            for field_token in field_tokens:
                if abs(len(field_token) - len(query_token)) > 5:
                    continue
                score = float(fuzz.ratio(query_token, field_token))
                if score > best_score:
                    best_score = score
                    best_token = field_token
            if best_score >= 91:
                hits.append((query_token, best_token, best_score))
        return hits

    def _score_row(
        self,
        query: str,
        query_tokens: Sequence[str],
        query_concepts: Set[str],
        indexed_row: Dict[str, object],
        include_broad_matches: bool,
    ) -> Tuple[float, List[str], str, bool]:
        query_normalized = self._normalize(query)
        total_score = 0.0
        matched_fields: List[str] = []
        reasons: List[str] = []
        strong_signal = False

        for column in self.search_columns:
            text = indexed_row["field_text"][column]
            if not text:
                continue

            field_tokens = indexed_row["field_tokens"][column]
            weights = FIELD_WEIGHTS.get(column, {})
            field_score = 0.0
            field_reason_parts: List[str] = []

            if query_normalized and len(query_normalized) >= 4 and query_normalized in text:
                field_score += weights.get("phrase", 0.0)
                field_reason_parts.append(f"phrase match in {FIELD_LABELS.get(column, column)}")
                strong_signal = True

            overlap = [token for token in query_tokens if token in field_tokens]
            if overlap:
                token_score = sum(self._idf.get(token, 1.0) for token in overlap) * weights.get("token", 0.0)
                field_score += token_score
                field_reason_parts.append(f"{FIELD_LABELS.get(column, column)} matched {', '.join(sorted(set(overlap)))}")
                strong_signal = True

            prefix_hits = []
            if weights.get("prefix", 0.0):
                for query_token in query_tokens:
                    if len(query_token) < 3 or query_token in overlap:
                        continue
                    if any(field_token.startswith(query_token) for field_token in field_tokens):
                        prefix_hits.append(query_token)
                if prefix_hits:
                    field_score += len(prefix_hits) * weights.get("prefix", 0.0)
                    field_reason_parts.append(
                        f"{FIELD_LABELS.get(column, column)} starts with {', '.join(sorted(set(prefix_hits)))}"
                    )
                    strong_signal = True

            fuzzy_hits = self._best_fuzzy_hits(query_tokens, field_tokens, include_broad_matches)
            if fuzzy_hits:
                field_score += sum((score / 100.0) * weights.get("fuzzy", 0.0) for _, _, score in fuzzy_hits)
                hit_labels = [f"{query_token}->{field_token}" for query_token, field_token, _ in fuzzy_hits[:2]]
                field_reason_parts.append(f"close spelling match {', '.join(hit_labels)}")

            if field_score > 0:
                total_score += field_score
                matched_fields.append(FIELD_LABELS.get(column, column))
                if field_reason_parts:
                    reasons.append(field_reason_parts[0])

        shared_concepts = query_concepts & indexed_row["concepts"]
        if shared_concepts:
            total_score += 18.0 * len(shared_concepts)
            reasons.insert(0, f"topic match on {', '.join(sorted(shared_concepts))}")
            strong_signal = True

        unique_reasons = []
        seen_reasons = set()
        for reason in reasons:
            if reason not in seen_reasons:
                unique_reasons.append(reason)
                seen_reasons.add(reason)

        return total_score, matched_fields, "; ".join(unique_reasons[:3]), strong_signal

    def _matches_dataset_filter(
        self,
        indexed_row: Dict[str, object],
        dataset_filter: Optional[Union[str, Sequence[str]]],
    ) -> bool:
        if not dataset_filter:
            return True

        targets = dataset_filter if isinstance(dataset_filter, Sequence) and not isinstance(dataset_filter, str) else [dataset_filter]
        normalized_targets = [self._normalize(value) for value in targets if str(value).strip()]
        if not normalized_targets:
            return True

        for target in normalized_targets:
            for dataset_key in indexed_row["normalized_dataset_keys"]:
                if target == dataset_key or target in dataset_key or dataset_key in target:
                    return True
        return False

    def _matches_topic_filter(
        self,
        row: pd.Series,
        topic_filter: Optional[Union[str, Sequence[str]]],
    ) -> bool:
        if not topic_filter:
            return True

        row_topics = [self._normalize(topic) for topic in self._split_terms(row["topic"])]
        targets = topic_filter if isinstance(topic_filter, Sequence) and not isinstance(topic_filter, str) else [topic_filter]
        normalized_targets = [self._normalize(value) for value in targets if str(value).strip()]
        return any(target in row_topics for target in normalized_targets)

    def _matches_publication_type_filter(
        self,
        row: pd.Series,
        publication_type_filter: Optional[Union[str, Sequence[str]]],
    ) -> bool:
        if not publication_type_filter:
            return True

        targets = publication_type_filter if isinstance(publication_type_filter, Sequence) and not isinstance(publication_type_filter, str) else [publication_type_filter]
        normalized_targets = {self._normalize(value) for value in targets if str(value).strip()}
        return self._normalize(row["publication_type"]) in normalized_targets

    @staticmethod
    def _matches_year_range(row: pd.Series, year_range: Optional[Tuple[int, int]]) -> bool:
        if not year_range:
            return True
        start_year, end_year = year_range
        year = int(row["year"])
        return start_year <= year <= end_year

    def catalog(
        self,
        dataset_filter: Optional[Union[str, Sequence[str]]] = None,
        topic_filter: Optional[Union[str, Sequence[str]]] = None,
        publication_type_filter: Optional[Union[str, Sequence[str]]] = None,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        selected_indexes = []
        for index, (_, row) in enumerate(self.df.iterrows()):
            indexed_row = self._indexed_rows[index]
            if not self._matches_dataset_filter(indexed_row, dataset_filter):
                continue
            if not self._matches_topic_filter(row, topic_filter):
                continue
            if not self._matches_publication_type_filter(row, publication_type_filter):
                continue
            if not self._matches_year_range(row, year_range):
                continue
            selected_indexes.append(index)

        if not selected_indexes:
            return self.df.head(0)[self.public_columns].copy()

        frame = self.df.iloc[selected_indexes][self.public_columns].copy()
        return frame.sort_values(["year", "paper_title"], ascending=[False, True]).reset_index(drop=True)

    def featured(self, limit: int = 8) -> pd.DataFrame:
        priorities = {paper_title: index for index, paper_title in enumerate(FEATURED_PAPERS)}
        featured = self.df[self.df["paper_title"].isin(priorities)].copy()
        if featured.empty:
            return self.df[self.public_columns].sort_values(["year", "paper_title"], ascending=[False, True]).head(limit)

        featured["_priority"] = featured["paper_title"].map(priorities)
        featured = featured.sort_values(["_priority", "year"], ascending=[True, False]).drop(columns="_priority")
        return featured[self.public_columns].head(limit).reset_index(drop=True)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 30.0,
        dataset_filter: Optional[Union[str, Sequence[str]]] = None,
        topic_filter: Optional[Union[str, Sequence[str]]] = None,
        publication_type_filter: Optional[Union[str, Sequence[str]]] = None,
        year_range: Optional[Tuple[int, int]] = None,
        include_broad_matches: bool = False,
    ) -> pd.DataFrame:
        query_normalized = self._normalize(query)
        if not query_normalized:
            return self.df.head(0).copy()

        query_tokens = self._informative_query_tokens(query)
        query_concepts = self._extract_concepts(query_normalized)
        if not query_concepts:
            query_concepts = self._extract_concepts(" ".join(query_tokens))

        results: List[PaperSearchResult] = []
        fallback_rows = []

        for index, (_, row) in enumerate(self.df.iterrows()):
            indexed_row = self._indexed_rows[index]
            if not self._matches_dataset_filter(indexed_row, dataset_filter):
                continue
            if not self._matches_topic_filter(row, topic_filter):
                continue
            if not self._matches_publication_type_filter(row, publication_type_filter):
                continue
            if not self._matches_year_range(row, year_range):
                continue

            score, matched_fields, match_reason, strong_signal = self._score_row(
                query=query,
                query_tokens=query_tokens,
                query_concepts=query_concepts,
                indexed_row=indexed_row,
                include_broad_matches=include_broad_matches,
            )

            if include_broad_matches and score > 0:
                item = dict(indexed_row["row"])
                item["_score"] = round(score, 2)
                item["_matched_fields"] = ", ".join(matched_fields)
                item["_match_reason"] = match_reason
                item["_confidence"] = self._confidence_label(score)
                fallback_rows.append(item)

            if not include_broad_matches and not strong_signal:
                continue
            if score < min_score:
                continue

            results.append(
                PaperSearchResult(
                    row=dict(indexed_row["row"]),
                    score=score,
                    matched_fields=matched_fields,
                    match_reason=match_reason,
                    confidence=self._confidence_label(score),
                )
            )

        if not results and include_broad_matches:
            if not fallback_rows:
                return self.df.head(0).copy()
            return (
                pd.DataFrame(fallback_rows)
                .sort_values(["_score", "year", "paper_title"], ascending=[False, False, True])
                .head(limit)
                .reset_index(drop=True)
            )

        if not results:
            return self.df.head(0).copy()

        output_rows = []
        for result in results:
            item = dict(result.row)
            item["_score"] = round(result.score, 2)
            item["_matched_fields"] = ", ".join(result.matched_fields)
            item["_match_reason"] = result.match_reason
            item["_confidence"] = result.confidence
            output_rows.append(item)

        return (
            pd.DataFrame(output_rows)
            .sort_values(["_score", "year", "paper_title"], ascending=[False, False, True])
            .head(limit)
            .reset_index(drop=True)
        )

    def related_to_datasets(self, dataset_names: Sequence[str], limit: int = 3) -> pd.DataFrame:
        normalized_targets = [self._normalize(name) for name in dataset_names if self._normalize(name)]
        if not normalized_targets:
            return self.df.head(0).copy()

        rows = []
        for indexed_row in self._indexed_rows:
            overlaps = []
            for target in normalized_targets:
                for dataset_key in indexed_row["normalized_dataset_keys"]:
                    if target == dataset_key or target in dataset_key or dataset_key in target:
                        overlaps.append(dataset_key)
            if not overlaps:
                continue

            item = dict(indexed_row["row"])
            item["_related_score"] = len(set(overlaps)) * 100 + int(item["year"])
            item["_matched_datasets"] = ", ".join(indexed_row["dataset_keys"])
            rows.append(item)

        if not rows:
            return self.df.head(0).copy()

        return (
            pd.DataFrame(rows)
            .sort_values(["_related_score", "year", "paper_title"], ascending=[False, False, True])
            .head(limit)
            .reset_index(drop=True)
        )

    def dataset_usage_summary(self, limit: Optional[int] = 12) -> pd.DataFrame:
        exploded_rows = []
        for _, row in self.df.sort_values(["year", "paper_title"], ascending=[False, True]).iterrows():
            for dataset_name in self._split_terms(row["dataset_keys"]):
                exploded_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "paper_title": row["paper_title"],
                        "year": int(row["year"]),
                        "publication_type": row["publication_type"],
                    }
                )

        if not exploded_rows:
            return pd.DataFrame(columns=["dataset_name", "paper_count", "latest_year", "example_paper"])

        exploded = pd.DataFrame(exploded_rows).sort_values(["dataset_name", "year", "paper_title"], ascending=[True, False, True])
        summary = (
            exploded.groupby("dataset_name", as_index=False)
            .agg(
                paper_count=("paper_title", "count"),
                latest_year=("year", "max"),
                example_paper=("paper_title", "first"),
            )
            .sort_values(["paper_count", "latest_year", "dataset_name"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        if limit is None:
            return summary
        return summary.head(limit)

    def suggest(self, query: str, limit: int = 8) -> List[str]:
        query_normalized = self._normalize(query)
        if not query_normalized:
            return list(self.featured(limit)["paper_title"].astype(str))

        suggestions = []
        search_results = self.search(
            query=query,
            limit=limit,
            min_score=18.0,
            include_broad_matches=True,
        )
        if not search_results.empty:
            suggestions.extend(search_results["paper_title"].astype(str).tolist())

        for dataset_name in self.dataset_options:
            if query_normalized in self._normalize(dataset_name):
                suggestions.append(dataset_name)

        for topic_name in self.topic_options:
            if query_normalized in self._normalize(topic_name):
                suggestions.append(topic_name)

        ordered = []
        seen = set()
        for suggestion in suggestions:
            normalized = self._normalize(suggestion)
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(suggestion)
        return ordered[:limit]

    @staticmethod
    def highlight_text(text: str, query: str) -> str:
        if not text:
            return ""

        safe_text = html.escape(str(text))
        tokens = PapersCatalog._tokenize(query)
        if not tokens:
            return safe_text

        highlighted = safe_text
        for token in sorted(set(tokens), key=len, reverse=True):
            pattern = re.compile(rf"(?i)\b{re.escape(token)}\b")
            highlighted = pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", highlighted)
        return highlighted
