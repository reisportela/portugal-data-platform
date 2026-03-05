import streamlit as st
import pandas as pd
import plotly.express as px

from modules.search_engine import SearchEngine
from modules.econometrics import run_ols
from modules.observatory import fake_indicator
from modules.export import export_csv

st.set_page_config(page_title="Portugal Data Platform", layout="wide")

st.title("Portugal Data Platform")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Dataset Search",
        "Data Lab",
        "Economic Observatory",
    ],
)

if page == "Home":
    st.write(
        """
Welcome to the Portugal Data Platform.

This portal helps discover Portuguese datasets,
run simple econometric analyses, and explore economic indicators.
"""
    )

elif page == "Dataset Search":
    st.header("Dataset Search")

    engine = SearchEngine("dataset_catalog.csv")

    with st.sidebar:
        st.subheader("Search filters")
        domain_filter = st.text_input("Domain contains")
        institution_filter = st.text_input("Institution contains")

    query = st.text_input("Search dataset")

    if query:
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
                st.write("Institution:", r["institution"])
                st.write("Domain:", r["domain"])

                if "unit" in results.columns and str(r.get("unit", "")).strip():
                    st.write("Unit:", r["unit"])

                if "spatial_level" in results.columns and str(r.get("spatial_level", "")).strip():
                    st.write("Spatial level:", r["spatial_level"])

                if "time_coverage" in results.columns and str(r.get("time_coverage", "")).strip():
                    st.write("Time coverage:", r["time_coverage"])

                if "access" in results.columns and str(r.get("access", "")).strip():
                    st.write("Access:", r["access"])

                if "_score" in results.columns:
                    st.write("Score:", r["_score"])

                if "_matched_fields" in results.columns and str(r["_matched_fields"]).strip():
                    st.write("Matched fields:", r["_matched_fields"])

                st.markdown(f"[Source]({r['link']})")
                st.divider()

elif page == "Data Lab":
    st.header("Econometrics Data Lab")

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        y = st.selectbox("Dependent variable", df.columns)
        X = st.multiselect("Independent variables", df.columns)

        if st.button("Run OLS"):
            if not X:
                st.warning("Select at least one independent variable.")
            else:
                try:
                    model = run_ols(df, y, X)
                    st.text(model.summary())
                    st.download_button("Download CSV", export_csv(df), "data.csv")
                except Exception as e:
                    st.error(f"Model failed: {e}")

elif page == "Economic Observatory":
    st.header("Economic Observatory")

    df = fake_indicator()
    fig = px.line(df, x="year", y="value", title="Unemployment Rate (example)")
    st.plotly_chart(fig, use_container_width=True)
