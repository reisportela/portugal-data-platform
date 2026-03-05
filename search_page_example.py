import streamlit as st
from modules.search_engine import SearchEngine

st.set_page_config(page_title="Dataset Search", layout="wide")

st.title("Dataset Search")

engine = SearchEngine("dataset_catalog.csv")

with st.sidebar:
    st.header("Filters")
    domain_filter = st.text_input("Domain contains")
    institution_filter = st.text_input("Institution contains")

query = st.text_input("Search datasets", placeholder="e.g. wages, labour, innovation, INE")

if query:
    suggestions = engine.suggest(query)
    if suggestions:
        st.caption("Suggestions: " + " | ".join(suggestions))

    results = engine.search(
        query=query,
        limit=15,
        domain_filter=domain_filter or None,
        institution_filter=institution_filter or None,
    )

    if results.empty:
        st.warning("No datasets found.")
    else:
        st.caption(f"{len(results)} result(s)")
        for _, r in results.iterrows():
            st.subheader(r["dataset_name"])
            st.write("**Institution:**", r["institution"])
            st.write("**Domain:**", r["domain"])

            if "unit" in results.columns and r.get("unit", ""):
                st.write("**Unit:**", r["unit"])
            if "spatial_level" in results.columns and r.get("spatial_level", ""):
                st.write("**Spatial level:**", r["spatial_level"])
            if "time_coverage" in results.columns and r.get("time_coverage", ""):
                st.write("**Time coverage:**", r["time_coverage"])
            if "access" in results.columns and r.get("access", ""):
                st.write("**Access:**", r["access"])

            st.write("**Score:**", r["_score"])
            if r["_matched_fields"]:
                st.write("**Matched fields:**", r["_matched_fields"])

            st.markdown(f"[Open source]({r['link']})")
            st.divider()
