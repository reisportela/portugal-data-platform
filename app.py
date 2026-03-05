
import streamlit as st
import pandas as pd
from modules.search_engine import SearchEngine
from modules.econometrics import run_ols
from modules.observatory import fake_indicator
from modules.export import export_csv
import plotly.express as px

st.set_page_config(page_title="Portugal Data Platform",layout="wide")

st.title("Portugal Data Platform")

page = st.sidebar.selectbox("Navigation",[
"Home",
"Dataset Search",
"Data Lab",
"Economic Observatory"
])

if page=="Home":
    st.write("""
Welcome to the Portugal Data Platform.

This portal helps discover Portuguese datasets,
run simple econometric analyses, and explore economic indicators.
""")

elif page=="Dataset Search":
    st.header("Dataset Search")
    engine = SearchEngine("dataset_catalog.csv")
    query = st.text_input("Search dataset")
    if query:
        results = engine.search(query)
        for r in results:
            st.subheader(r["dataset_name"])
            st.write("Institution:",r["institution"])
            st.write("Domain:",r["domain"])
            st.markdown(f"[Source]({r['link']})")
            st.divider()

elif page=="Data Lab":
    st.header("Econometrics Data Lab")
    uploaded = st.file_uploader("Upload CSV dataset")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        y = st.selectbox("Dependent variable",df.columns)
        X = st.multiselect("Independent variables",df.columns)
        if st.button("Run OLS") and X:
            model = run_ols(df,y,X)
            st.text(model.summary())
            st.download_button("Download CSV",export_csv(df),"data.csv")

elif page=="Economic Observatory":
    st.header("Economic Observatory")
    df = fake_indicator()
    fig = px.line(df,x="year",y="value",title="Unemployment Rate (example)")
    st.plotly_chart(fig)
