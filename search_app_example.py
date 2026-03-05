
import streamlit as st
import pandas as pd
from modules.search_engine import SearchEngine

st.title("Dataset Search")

engine = SearchEngine("dataset_catalog.csv")

query = st.text_input("Search dataset")

if query:

    results = engine.search(query)

    if len(results) == 0:
        st.warning("No datasets found")

    else:

        for _,r in results.iterrows():
            st.subheader(r["dataset_name"])
            st.write("Institution:", r["institution"])
            st.write("Domain:", r["domain"])
            st.markdown(f"[Source]({r['link']})")
            st.divider()
