# Full Working Search Engine

Files included:

- `modules/search_engine.py` → production-ready search component
- `search_page_example.py` → Streamlit example page

## How to use

1. Replace your current `modules/search_engine.py` with this version.
2. Keep `dataset_catalog.csv` in your project root.
3. Import in Streamlit:

```python
from modules.search_engine import SearchEngine
engine = SearchEngine("dataset_catalog.csv")
results = engine.search("wages")
```

## Features

- ranked results
- search across dataset name, institution, domain, and optional metadata columns
- suggestions
- filter by domain/institution
- matched-fields output
- simple HTML highlighter helper
