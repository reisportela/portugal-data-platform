import html
import math
from pathlib import Path
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
from rapidfuzz import fuzz


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "da",
    "das",
    "data",
    "dataset",
    "datasets",
    "de",
    "do",
    "dos",
    "for",
    "in",
    "indicator",
    "indicators",
    "index",
    "market",
    "of",
    "on",
    "or",
    "rates",
    "series",
    "source",
    "sources",
    "statistics",
    "survey",
    "table",
    "tables",
    "the",
}

FIELD_LABELS = {
    "dataset_name": "title",
    "domain": "domain",
    "institution": "institution",
    "description": "description",
    "keywords": "keywords",
    "aliases": "topic tags",
    "source_type": "source type",
}

FIELD_WEIGHTS = {
    "dataset_name": {"phrase": 60.0, "token": 16.0, "prefix": 9.0, "fuzzy": 8.0},
    "keywords": {"phrase": 42.0, "token": 13.0, "prefix": 7.0, "fuzzy": 5.0},
    "aliases": {"phrase": 36.0, "token": 11.0, "prefix": 6.0, "fuzzy": 4.0},
    "description": {"phrase": 22.0, "token": 7.5, "prefix": 0.0, "fuzzy": 0.0},
    "domain": {"phrase": 26.0, "token": 8.0, "prefix": 6.0, "fuzzy": 4.0},
    "institution": {"phrase": 20.0, "token": 6.0, "prefix": 4.0, "fuzzy": 3.0},
    "source_type": {"phrase": 10.0, "token": 4.0, "prefix": 0.0, "fuzzy": 0.0},
}

DOMAIN_KEYWORDS = {
    "agriculture": ["agriculture", "farming", "crops", "rural economy"],
    "climate": ["climate", "weather", "temperature", "precipitation"],
    "construction": ["construction", "building", "permits", "housing supply"],
    "consumption": ["consumption", "household expenditure", "spending", "budget"],
    "crime": ["crime", "offences", "public safety", "criminality"],
    "demography": ["population", "demography", "households", "census"],
    "digital economy": ["digital economy", "internet", "ict", "broadband", "e-commerce"],
    "education": ["education", "schools", "students", "higher education"],
    "energy": ["energy", "electricity", "renewables", "energy consumption"],
    "entrepreneurship": ["entrepreneurship", "startups", "new firms", "venture creation"],
    "environment": ["environment", "pollution", "waste", "water", "emissions"],
    "finance": ["finance", "credit", "interest rates", "banking", "monetary"],
    "firms": ["firms", "companies", "businesses", "enterprises"],
    "fisheries": ["fisheries", "fish", "aquaculture", "marine economy"],
    "gender": ["gender", "women", "men", "equality", "pay gap"],
    "health": ["health", "hospitals", "mortality", "healthcare"],
    "housing": ["housing", "house prices", "rents", "dwellings", "real estate"],
    "income": ["income", "poverty", "inequality", "living conditions"],
    "innovation": ["innovation", "r&d", "patents", "technology"],
    "justice": ["justice", "courts", "judicial system", "legal cases"],
    "labour": ["labour market", "employment", "jobs", "workers"],
    "macroeconomics": ["gdp", "growth", "output", "value added", "national accounts"],
    "migration": ["migration", "immigration", "emigration", "foreign population"],
    "prices": ["inflation", "consumer prices", "producer prices", "price index"],
    "productivity": ["productivity", "output per worker", "efficiency", "tfp"],
    "public finance": ["public finance", "taxes", "public debt", "government budget"],
    "regional": ["regional", "municipal", "local", "territorial indicators"],
    "tourism": ["tourism", "accommodation", "visitors", "overnight stays"],
    "trade": ["trade", "exports", "imports", "foreign trade"],
    "transport": ["transport", "mobility", "road", "air", "ports"],
}

INSTITUTION_KEYWORDS = {
    "AT": ["tax authority", "fiscal administration"],
    "Banco de Portugal": ["banco de portugal", "central bank", "bpstat"],
    "CIG": ["cig"],
    "CMVM": ["cmvm"],
    "DGEEC": ["dgeec"],
    "DGEG": ["dgeg"],
    "DGPJ": ["dgpj"],
    "Euronext Lisbon": ["euronext lisbon", "stock exchange"],
    "GEP": ["gep"],
    "IGCP": ["igcp", "treasury agency"],
    "IMT": ["imt"],
    "INE": ["ine", "official statistics", "statistics portugal"],
    "IPMA": ["ipma"],
    "PORDATA": ["pordata"],
    "SEF": ["sef"],
    "SNS": ["sns"],
    "Startup Portugal": ["startup portugal"],
}

CONCEPT_GROUPS = {
    "agriculture": {"agriculture", "agricultura", "crop", "crops", "farming"},
    "climate": {
        "clima",
        "climate",
        "meteorology",
        "precipitacao",
        "precipitation",
        "temperatura",
        "temperature",
        "weather",
    },
    "consumption": {
        "budget",
        "consumption",
        "consumo",
        "despesa",
        "expenditure",
        "gastos",
        "spending",
    },
    "crime": {
        "crime",
        "criminalidade",
        "criminality",
        "offence",
        "offences",
        "safety",
        "seguranca",
    },
    "demography": {
        "age structure",
        "census",
        "censos",
        "demografia",
        "demography",
        "households",
        "population",
        "populacao",
    },
    "digital": {
        "broadband",
        "digital",
        "digital economy",
        "e-commerce",
        "ict",
        "informatica",
        "internet",
    },
    "education": {
        "alunos",
        "educacao",
        "education",
        "escola",
        "escolas",
        "higher education",
        "school",
        "schools",
        "students",
        "university",
        "universidade",
    },
    "energy": {
        "electricidade",
        "electricity",
        "energia",
        "energy",
        "energy consumption",
        "renewable",
        "renewables",
    },
    "environment": {
        "agua",
        "ambiente",
        "emissions",
        "environment",
        "pollution",
        "residuos",
        "waste",
        "water",
    },
    "finance": {
        "bank",
        "banking",
        "banks",
        "credito",
        "credit",
        "financial",
        "finance",
        "interest rates",
        "juros",
        "loans",
        "monetary",
    },
    "firms": {
        "business",
        "businesses",
        "companies",
        "company",
        "empresas",
        "enterprise",
        "enterprises",
        "establishment",
        "establishments",
        "firm",
        "firms",
    },
    "gender": {
        "equality",
        "female",
        "gender",
        "genero",
        "igualdade",
        "male",
        "men",
        "women",
    },
    "health": {
        "health",
        "healthcare",
        "hospital",
        "hospitals",
        "morbidity",
        "mortality",
        "saude",
    },
    "housing": {
        "casa",
        "casas",
        "dwelling",
        "dwellings",
        "habitacao",
        "homes",
        "house prices",
        "housing",
        "imobiliario",
        "property prices",
        "real estate",
        "rent",
        "rents",
    },
    "income": {
        "deprivation",
        "desigualdade",
        "household income",
        "income",
        "inequality",
        "living conditions",
        "pobreza",
        "poverty",
        "rendimento",
        "rendimentos",
    },
    "inflation": {
        "consumer prices",
        "cost of living",
        "cpi",
        "inflacao",
        "inflation",
        "ppi",
        "precos",
        "price index",
        "producer prices",
    },
    "innovation": {
        "desenvolvimento",
        "inovacao",
        "innovation",
        "patent",
        "patents",
        "r&d",
        "research and development",
        "startup",
        "technology",
        "venture capital",
    },
    "justice": {
        "courts",
        "judicial",
        "justice",
        "justica",
        "legal",
        "tribunal",
        "tribunals",
    },
    "labour": {
        "desemprego",
        "emprego",
        "employment",
        "job",
        "joblessness",
        "jobs",
        "labor",
        "labor market",
        "labour",
        "labour market",
        "mercado de trabalho",
        "trabalho",
        "trabalhadores",
        "unemployment",
        "worker",
        "workers",
    },
    "macro": {
        "contas nacionais",
        "economic activity",
        "economy",
        "gdp",
        "growth",
        "national accounts",
        "output",
        "pib",
        "regional accounts",
        "value added",
    },
    "migration": {
        "emigracao",
        "emigration",
        "foreign population",
        "imigracao",
        "immigration",
        "migrant",
        "migrants",
        "migration",
        "migracao",
    },
    "productivity": {
        "efficiency",
        "labour productivity",
        "output per worker",
        "productivity",
        "produtividade",
        "tfp",
    },
    "public_finance": {
        "divida publica",
        "fiscal",
        "government budget",
        "government finance",
        "impostos",
        "income tax",
        "orcamento",
        "public debt",
        "public finance",
        "tax",
        "tax revenue",
        "taxes",
        "vat",
    },
    "regional": {
        "concelho",
        "concelhos",
        "local",
        "municipal",
        "municipalities",
        "municipality",
        "municipio",
        "municipios",
        "nuts",
        "region",
        "regional",
        "regions",
        "territorial",
    },
    "tourism": {
        "accommodation",
        "hotel",
        "hotels",
        "overnight stays",
        "tourism",
        "tourist",
        "tourists",
        "turismo",
    },
    "trade": {
        "comercio",
        "exports",
        "foreign trade",
        "goods",
        "imports",
        "trade",
    },
    "transport": {
        "air transport",
        "mobility",
        "mobilidade",
        "portos",
        "ports",
        "public transport",
        "road",
        "transport",
        "transporte",
    },
    "wages": {
        "earnings",
        "payroll",
        "remuneracao",
        "salary",
        "salaries",
        "salario",
        "salarios",
        "wage",
        "wages",
    },
}

CONCEPT_LABELS = {
    "digital": "digital economy",
    "macro": "macroeconomics",
    "public_finance": "public finance",
}

FEATURED_DATASETS = [
    "Quadros de Pessoal",
    "Labour Force Survey",
    "Unemployment Statistics",
    "EU-SILC",
    "Central Balance Sheet Database",
    "National Accounts",
    "Price Index (CPI)",
    "Municipal Indicators",
]


@dataclass
class SearchResult:
    row: Dict[str, object]
    score: float
    matched_fields: List[str]
    match_reason: str
    confidence: str


CURATED_DATASET_METADATA = {
    "Quadros de Pessoal": {
        "description": "Matched employer-employee administrative data on workers establishments contracts occupations and wages in Portugal.",
        "keywords": [
            "wages",
            "earnings",
            "salaries",
            "salarios",
            "payroll",
            "workers",
            "employees",
            "firms",
            "establishments",
            "occupations",
            "contracts",
        ],
        "source_type": "Administrative data",
    },
    "EU-SILC": {
        "description": "Household microdata on income poverty inequality deprivation and living conditions.",
        "keywords": [
            "income",
            "poverty",
            "inequality",
            "living conditions",
            "households",
            "earnings",
            "social transfers",
        ],
        "source_type": "Survey",
    },
    "Labour Force Survey": {
        "description": "Official labour force survey covering employment unemployment participation occupations and hours worked.",
        "keywords": [
            "employment",
            "unemployment",
            "joblessness",
            "labour market",
            "participation",
            "workers",
            "hours worked",
        ],
        "source_type": "Survey",
    },
    "Central Balance Sheet Database": {
        "description": "Firm-level financial statement database for Portuguese non-financial corporations.",
        "keywords": [
            "firms",
            "companies",
            "balance sheets",
            "financial statements",
            "investment",
            "leverage",
            "productivity",
            "corporate finance",
        ],
        "source_type": "Database",
    },
    "Community Innovation Survey": {
        "description": "Survey on firm innovation activities including product process and organizational innovation.",
        "keywords": [
            "innovation",
            "r&d",
            "technology",
            "firms",
            "patents",
            "product innovation",
            "process innovation",
        ],
        "source_type": "Survey",
    },
    "Household Budget Survey": {
        "description": "Survey on household expenditure consumption baskets and spending patterns.",
        "keywords": [
            "consumption",
            "household expenditure",
            "spending",
            "budget",
            "living standards",
        ],
        "source_type": "Survey",
    },
    "National Accounts": {
        "description": "Core macroeconomic accounts covering GDP output expenditure income and value added.",
        "keywords": [
            "gdp",
            "macroeconomics",
            "growth",
            "output",
            "value added",
            "income accounts",
        ],
        "source_type": "Accounts",
    },
    "Regional Accounts": {
        "description": "Regional macroeconomic accounts with GDP output and value added by territory.",
        "keywords": [
            "regional gdp",
            "regional output",
            "value added",
            "nuts",
            "territorial accounts",
        ],
        "source_type": "Accounts",
    },
    "Population Census": {
        "description": "Census data on population households dwellings and demographic structure.",
        "keywords": [
            "population",
            "households",
            "dwellings",
            "demography",
            "census",
            "municipal population",
        ],
        "source_type": "Census",
    },
    "Employment Statistics": {
        "description": "Statistical series on employment jobs workers and labour market conditions.",
        "keywords": [
            "employment",
            "jobs",
            "workers",
            "labour market",
            "workforce",
        ],
        "source_type": "Statistical series",
    },
    "Unemployment Statistics": {
        "description": "Statistical series on unemployment joblessness unemployment rates and job seekers.",
        "keywords": [
            "unemployment",
            "joblessness",
            "desemprego",
            "unemployment rate",
            "job seekers",
            "labour market",
        ],
        "source_type": "Statistical series",
    },
    "Price Index (CPI)": {
        "description": "Consumer price index series used to track inflation and cost of living.",
        "keywords": [
            "inflation",
            "consumer prices",
            "cpi",
            "cost of living",
            "precos",
            "inflacao",
        ],
        "source_type": "Index",
    },
    "Producer Price Index": {
        "description": "Producer price index series for producer cost pressures and pipeline inflation.",
        "keywords": [
            "producer prices",
            "ppi",
            "inflation",
            "industrial prices",
        ],
        "source_type": "Index",
    },
    "Regional Development Indicators": {
        "description": "Regional indicator set covering local socioeconomic development and territorial disparities.",
        "keywords": [
            "regional",
            "municipal",
            "local indicators",
            "territorial development",
            "regions",
        ],
        "source_type": "Indicator set",
    },
    "Municipal Indicators": {
        "description": "Municipal-level indicator set for local socioeconomic comparisons across Portugal.",
        "keywords": [
            "municipal",
            "municipality",
            "concelho",
            "local indicators",
            "regional comparisons",
        ],
        "source_type": "Indicator set",
    },
    "Business Demography": {
        "description": "Statistics on firm births deaths survival and business population dynamics.",
        "keywords": [
            "firms",
            "companies",
            "firm births",
            "firm deaths",
            "survival",
            "entrepreneurship",
        ],
        "source_type": "Statistical series",
    },
    "Foreign Trade Statistics": {
        "description": "Trade statistics on exports imports and foreign trade flows.",
        "keywords": [
            "trade",
            "exports",
            "imports",
            "goods trade",
            "external sector",
        ],
        "source_type": "Statistical series",
    },
    "Financial Accounts": {
        "description": "Financial accounts covering sector balance sheets assets liabilities and funding positions.",
        "keywords": [
            "financial accounts",
            "assets",
            "liabilities",
            "balance sheets",
            "financial sector",
        ],
        "source_type": "Accounts",
    },
    "Monetary Statistics": {
        "description": "Monetary and banking statistics on money credit deposits and balance sheet aggregates.",
        "keywords": [
            "money supply",
            "banking",
            "credit",
            "deposits",
            "monetary policy",
        ],
        "source_type": "Statistical series",
    },
    "Interest Rates": {
        "description": "Interest rate statistics for loans deposits and financial market benchmarks.",
        "keywords": [
            "interest rates",
            "loan rates",
            "deposit rates",
            "monetary policy",
            "juros",
        ],
        "source_type": "Statistical series",
    },
    "Credit Statistics": {
        "description": "Statistics on credit flows loan stocks borrowers and lending conditions.",
        "keywords": [
            "credit",
            "loans",
            "borrowing",
            "household credit",
            "firm credit",
        ],
        "source_type": "Statistical series",
    },
    "Housing Statistics": {
        "description": "Statistics on dwellings housing stock residential conditions and households.",
        "keywords": [
            "housing",
            "dwellings",
            "homes",
            "residential stock",
            "households",
        ],
        "source_type": "Statistical series",
    },
    "Real Estate Prices": {
        "description": "Series on house prices residential property prices and real estate valuation.",
        "keywords": [
            "house prices",
            "property prices",
            "housing",
            "real estate",
            "imobiliario",
        ],
        "source_type": "Statistical series",
    },
    "Migration Statistics": {
        "description": "Migration statistics covering inflows outflows immigrants emigrants and foreign residents.",
        "keywords": [
            "migration",
            "immigration",
            "emigration",
            "migrants",
            "foreign population",
        ],
        "source_type": "Statistical series",
    },
    "Digital Economy Statistics": {
        "description": "Statistics on ICT use digital adoption internet access and online activity.",
        "keywords": [
            "digital economy",
            "ict",
            "internet",
            "digitalization",
            "e-commerce",
        ],
        "source_type": "Statistical series",
    },
    "Research and Development Survey": {
        "description": "Survey on R&D expenditure researchers research activity and innovation inputs.",
        "keywords": [
            "r&d",
            "research and development",
            "innovation",
            "researchers",
            "science",
        ],
        "source_type": "Survey",
    },
    "Regional Labour Statistics": {
        "description": "Regional labour market indicators including employment unemployment and labour market conditions by territory.",
        "keywords": [
            "regional labour",
            "regional employment",
            "regional unemployment",
            "wages",
            "municipal labour market",
        ],
        "source_type": "Statistical series",
    },
    "Gender Equality Statistics": {
        "description": "Indicators on gender gaps equality representation and labour market outcomes for women and men.",
        "keywords": [
            "gender equality",
            "women",
            "men",
            "pay gap",
            "representation",
        ],
        "source_type": "Indicator set",
    },
    "Climate Indicators": {
        "description": "Climate indicators on temperature precipitation drought and other weather conditions.",
        "keywords": [
            "climate",
            "temperature",
            "precipitation",
            "weather",
            "drought",
        ],
        "source_type": "Indicator set",
    },
    "Regional GDP": {
        "description": "Regional GDP series for territorial output and economic performance comparisons.",
        "keywords": [
            "regional gdp",
            "gdp",
            "regional output",
            "territorial growth",
        ],
        "source_type": "Statistical series",
    },
    "Productivity Statistics": {
        "description": "Statistics on labour productivity output per worker and efficiency.",
        "keywords": [
            "productivity",
            "labour productivity",
            "output per worker",
            "efficiency",
        ],
        "source_type": "Statistical series",
    },
}


class SearchEngine:
    """
    Search engine for the Portugal Data Platform.

    Core required columns:
    - dataset_name
    - institution
    - domain
    - link

    Optional columns are preserved and can be displayed by the UI.
    """

    def __init__(self, file_path: str):
        catalog_path = Path(file_path)
        frames = [pd.read_csv(catalog_path).fillna("")]
        extensions_path = catalog_path.parent / "data" / "search_catalog_extensions.csv"
        if extensions_path.exists():
            frames.append(pd.read_csv(extensions_path).fillna(""))

        self.df = pd.concat(frames, ignore_index=True, sort=False).fillna("")
        self.df = self.df.drop_duplicates(subset=["dataset_name", "institution", "link"], keep="first")
        self.required_columns = ["dataset_name", "institution", "domain", "link"]
        missing = [column for column in self.required_columns if column not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self._enrich_catalog()
        self.search_columns = [
            column
            for column in [
                "dataset_name",
                "institution",
                "domain",
                "description",
                "keywords",
                "aliases",
                "source_type",
                "unit",
                "spatial_level",
                "time_coverage",
                "access",
            ]
            if column in self.df.columns
        ]
        self.public_columns = [
            column
            for column in [
                "dataset_name",
                "institution",
                "domain",
                "source_type",
                "description",
                "keywords",
                "unit",
                "spatial_level",
                "time_coverage",
                "access",
                "link",
            ]
            if column in self.df.columns
        ]
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
        normalized = SearchEngine._normalize(text)
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
                normalized = SearchEngine._normalize(term)
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

    def _infer_source_type(self, dataset_name: str) -> str:
        name = self._normalize(dataset_name)
        if "survey" in name:
            return "Survey"
        if "census" in name:
            return "Census"
        if "database" in name:
            return "Database"
        if "accounts" in name:
            return "Accounts"
        if "indicators" in name:
            return "Indicator set"
        if "index" in name:
            return "Index"
        return "Statistical series"

    def _default_description(self, row: pd.Series, source_type: str) -> str:
        domain = str(row["domain"]).strip()
        institution = str(row["institution"]).strip()
        return f"{source_type} on {domain} published by {institution}."

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
        descriptions = []
        keywords_list = []
        aliases_list = []
        source_types = []
        concepts_list = []

        for _, row in self.df.iterrows():
            dataset_name = str(row["dataset_name"])
            institution = str(row["institution"])
            domain = str(row["domain"])
            curated = CURATED_DATASET_METADATA.get(dataset_name, {})

            source_type = str(row.get("source_type", "")).strip() or curated.get("source_type") or self._infer_source_type(dataset_name)
            description = str(row.get("description", "")).strip() or curated.get("description") or self._default_description(row, source_type)
            display_keywords = self._merge_terms(
                self._split_terms(row.get("keywords", "")),
                curated.get("keywords", []),
                DOMAIN_KEYWORDS.get(domain, []),
            )
            aliases = self._merge_terms(
                display_keywords,
                INSTITUTION_KEYWORDS.get(institution, []),
                [domain, institution, source_type],
            )
            concepts = self._merge_terms(
                self._extract_concepts(" ".join([dataset_name, description, " ".join(display_keywords), domain])),
                self._split_terms(row.get("concepts", "")),
            )

            descriptions.append(description)
            keywords_list.append(" | ".join(display_keywords))
            aliases_list.append(" | ".join(aliases))
            source_types.append(source_type)
            concepts_list.append(" | ".join(concepts))

        self.df["description"] = descriptions
        self.df["keywords"] = keywords_list
        self.df["aliases"] = aliases_list
        self.df["source_type"] = source_types
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

        concepts = set(self._split_terms(item.get("concepts", ""))) | self._extract_concepts(
            " ".join(field_text.values())
        )
        return {
            "row": item,
            "field_text": field_text,
            "field_tokens": field_tokens,
            "all_tokens": all_tokens,
            "concepts": concepts,
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
            concept_bonus = 18.0 * len(shared_concepts)
            total_score += concept_bonus
            concept_labels = [CONCEPT_LABELS.get(concept, concept.replace("_", " ")) for concept in sorted(shared_concepts)]
            reasons.insert(0, f"topic match on {', '.join(concept_labels)}")
            strong_signal = True

        unique_reasons = []
        seen_reasons = set()
        for reason in reasons:
            if reason not in seen_reasons:
                unique_reasons.append(reason)
                seen_reasons.add(reason)

        return total_score, matched_fields, "; ".join(unique_reasons[:3]), strong_signal

    def _apply_filter(
        self,
        frame: pd.DataFrame,
        column: str,
        filter_value: Optional[Union[str, Sequence[str]]],
    ) -> pd.DataFrame:
        if not filter_value:
            return frame

        if isinstance(filter_value, str):
            normalized = self._normalize(filter_value)
            return frame[
                frame[column].astype(str).map(lambda value: normalized in self._normalize(value))
            ]

        normalized_values = {self._normalize(value) for value in filter_value if str(value).strip()}
        if not normalized_values:
            return frame

        return frame[
            frame[column].astype(str).map(lambda value: self._normalize(value) in normalized_values)
        ]

    def catalog(
        self,
        domain_filter: Optional[Union[str, Sequence[str]]] = None,
        institution_filter: Optional[Union[str, Sequence[str]]] = None,
        source_type_filter: Optional[Union[str, Sequence[str]]] = None,
        access_filter: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        frame = self.df.copy()
        frame = self._apply_filter(frame, "domain", domain_filter)
        frame = self._apply_filter(frame, "institution", institution_filter)
        frame = self._apply_filter(frame, "source_type", source_type_filter)
        if "access" in frame.columns:
            frame = self._apply_filter(frame, "access", access_filter)
        return frame[self.public_columns].sort_values(["domain", "dataset_name"]).reset_index(drop=True)

    def featured(self, limit: int = 8) -> pd.DataFrame:
        priorities = {dataset_name: index for index, dataset_name in enumerate(FEATURED_DATASETS)}
        featured = self.df[self.df["dataset_name"].isin(priorities)].copy()
        if featured.empty:
            return self.df[self.public_columns].head(limit).reset_index(drop=True)

        featured["_priority"] = featured["dataset_name"].map(priorities)
        featured = featured.sort_values(["_priority", "dataset_name"]).drop(columns="_priority")
        return featured[self.public_columns].head(limit).reset_index(drop=True)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 35.0,
        domain_filter: Optional[Union[str, Sequence[str]]] = None,
        institution_filter: Optional[Union[str, Sequence[str]]] = None,
        source_type_filter: Optional[Union[str, Sequence[str]]] = None,
        access_filter: Optional[Union[str, Sequence[str]]] = None,
        include_broad_matches: bool = False,
    ) -> pd.DataFrame:
        query_normalized = self._normalize(query)
        if not query_normalized:
            return self.df.head(0).copy()

        query_tokens = self._informative_query_tokens(query)
        query_concepts = self._extract_concepts(query_normalized)
        if not query_concepts:
            query_concepts = self._extract_concepts(" ".join(query_tokens))

        working = self.df.copy()
        working = self._apply_filter(working, "domain", domain_filter)
        working = self._apply_filter(working, "institution", institution_filter)
        working = self._apply_filter(working, "source_type", source_type_filter)
        if "access" in working.columns:
            working = self._apply_filter(working, "access", access_filter)

        working_names = set(working["dataset_name"].astype(str))
        results: List[SearchResult] = []
        for indexed_row in self._indexed_rows:
            dataset_name = str(indexed_row["row"]["dataset_name"])
            if dataset_name not in working_names:
                continue

            score, matched_fields, match_reason, strong_signal = self._score_row(
                query=query,
                query_tokens=query_tokens,
                query_concepts=query_concepts,
                indexed_row=indexed_row,
                include_broad_matches=include_broad_matches,
            )

            if not include_broad_matches and not strong_signal:
                continue
            if score < min_score:
                continue

            row = dict(indexed_row["row"])
            results.append(
                SearchResult(
                    row=row,
                    score=score,
                    matched_fields=matched_fields,
                    match_reason=match_reason,
                    confidence=self._confidence_label(score),
                )
            )

        if not results and include_broad_matches:
            fallback_rows = []
            for indexed_row in self._indexed_rows:
                dataset_name = str(indexed_row["row"]["dataset_name"])
                if dataset_name not in working_names:
                    continue
                score, matched_fields, match_reason, _ = self._score_row(
                    query=query,
                    query_tokens=query_tokens,
                    query_concepts=query_concepts,
                    indexed_row=indexed_row,
                    include_broad_matches=True,
                )
                if score <= 0:
                    continue
                item = dict(indexed_row["row"])
                item["_score"] = round(score, 2)
                item["_matched_fields"] = ", ".join(matched_fields)
                item["_match_reason"] = match_reason
                item["_confidence"] = self._confidence_label(score)
                fallback_rows.append(item)
            if not fallback_rows:
                return self.df.head(0).copy()
            return pd.DataFrame(fallback_rows).sort_values("_score", ascending=False).head(limit).reset_index(drop=True)

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

        out = pd.DataFrame(output_rows).sort_values(
            by=["_score", "dataset_name"],
            ascending=[False, True],
        )
        return out.head(limit).reset_index(drop=True)

    def suggest(self, query: str, limit: int = 8) -> List[str]:
        query_normalized = self._normalize(query)
        if not query_normalized:
            return list(self.featured(limit)["dataset_name"].astype(str))

        suggestions = []
        search_results = self.search(
            query=query,
            limit=limit,
            min_score=20.0,
            include_broad_matches=True,
        )
        if not search_results.empty:
            suggestions.extend(search_results["dataset_name"].astype(str).tolist())

        for domain in sorted(self.df["domain"].astype(str).unique()):
            if query_normalized in self._normalize(domain):
                suggestions.append(domain)

        query_concepts = self._extract_concepts(query_normalized)
        concept_examples = {
            "finance": "interest rates",
            "firms": "business demography",
            "housing": "house prices",
            "income": "poverty",
            "inflation": "cpi",
            "labour": "unemployment",
            "macro": "gdp",
            "migration": "immigration",
            "regional": "municipal indicators",
            "wages": "wages",
        }
        for concept in sorted(query_concepts):
            example = concept_examples.get(concept)
            if example:
                suggestions.append(example)

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
        tokens = SearchEngine._tokenize(query)
        if not tokens:
            return safe_text

        highlighted = safe_text
        for token in sorted(set(tokens), key=len, reverse=True):
            pattern = re.compile(rf"(?i)\b{re.escape(token)}\b")
            highlighted = pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", highlighted)
        return highlighted
