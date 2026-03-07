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
- curated paper layer linked to datasets
- moderated source suggestion inbox

## Moderated Suggestions

The app now supports public source suggestions through a separate moderation inbox.

- Public submissions are stored in `data/source_suggestions.sqlite3`
- They do not appear in the public catalog automatically
- Approved suggestions still need manual promotion into `data/search_catalog_extensions.csv`

To enable the optional curator inbox in Streamlit, set a secret:

```toml
suggestions_admin_password = "your-password"
```
