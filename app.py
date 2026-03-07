import streamlit as st
import pandas as pd
import plotly.express as px

from modules.papers_catalog import PapersCatalog
from modules.search_engine import SearchEngine
from modules.econometrics import run_ols
from modules.observatory import fake_indicator
from modules.export import export_csv


APP_DATA_VERSION = "2026-03-07-2"


def safe_unique_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(value for value in frame[column].astype(str).unique() if value)


@st.cache_resource(show_spinner=False)
def load_search_engine(cache_version: str):
    return SearchEngine("dataset_catalog.csv")


@st.cache_resource(show_spinner=False)
def load_papers_catalog(cache_version: str):
    return PapersCatalog("data/papers_catalog.csv")


st.set_page_config(page_title="Portugal Data Platform", layout="wide")

st.title("Portugal Data Platform")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Dataset Search",
        "Research Papers",
        "Data Lab",
        "Economic Observatory",
    ],
)

if page == "Home":
    st.write(
        """
Welcome to the Portugal Data Platform.

This portal helps discover Portuguese datasets,
find papers that use them, run simple econometric analyses, and explore economic indicators.
"""
    )

elif page == "Dataset Search":
    st.header("Dataset Search")
    st.caption(
        "Search Portuguese datasets by topic, dataset name, source institution, or research concept. "
        "English and Portuguese queries are supported."
    )

    engine = load_search_engine(APP_DATA_VERSION)

    with st.sidebar:
        st.subheader("Search filters")
        domain_filter = st.multiselect(
            "Domain",
            safe_unique_values(engine.df, "domain"),
        )
        institution_filter = st.multiselect(
            "Institution",
            safe_unique_values(engine.df, "institution"),
        )
        source_type_filter = st.multiselect(
            "Source type",
            safe_unique_values(engine.df, "source_type"),
        )
        access_filter = st.multiselect(
            "Access",
            safe_unique_values(engine.df, "access"),
        )
        include_broad_matches = st.checkbox(
            "Include broader matches",
            value=False,
            help="Useful for vague or misspelled queries. Leave off for cleaner rankings.",
        )
        result_limit = st.slider("Maximum results", min_value=5, max_value=25, value=12)

    with st.expander("Search tips", expanded=False):
        st.write(
            "Try queries such as `unemployment`, `wages`, `inflation`, `firms`, "
            "`municipal data`, `house prices`, `immigration`, `desemprego`, "
            "`salarios`, or `inflacao`."
        )

    query = st.text_input(
        "Search datasets",
        placeholder="e.g. unemployment, wages, inflation, firm productivity, municipal data",
    )

    if query:
        papers_catalog = load_papers_catalog(APP_DATA_VERSION)
        suggestions = engine.suggest(query)
        if suggestions:
            st.caption("Suggestions: " + " | ".join(suggestions))

        results = engine.search(
            query=query,
            limit=result_limit,
            domain_filter=domain_filter or None,
            institution_filter=institution_filter or None,
            source_type_filter=source_type_filter or None,
            access_filter=access_filter or None,
            include_broad_matches=include_broad_matches,
        )

        if results.empty:
            st.warning("No high-confidence datasets found for this query.")
            if not include_broad_matches:
                st.info("Turn on `Include broader matches` in the sidebar if you want looser results.")
        else:
            st.caption(f"{len(results)} result(s) ranked by relevance")
            export_columns = [
                column
                for column in engine.public_columns + ["_score", "_confidence", "_match_reason"]
                if column in results.columns
            ]
            st.download_button(
                "Download results (CSV)",
                export_csv(results[export_columns]),
                "dataset_search_results.csv",
                mime="text/csv",
            )

            for _, r in results.iterrows():
                st.markdown(
                    f"### {engine.highlight_text(r['dataset_name'], query)}",
                    unsafe_allow_html=True,
                )
                meta_cols = st.columns(4)
                meta_cols[0].markdown(f"**Institution**  \n{r['institution']}")
                meta_cols[1].markdown(f"**Domain**  \n{r['domain']}")
                meta_cols[2].markdown(f"**Source type**  \n{r.get('source_type', '')}")
                meta_cols[3].markdown(f"**Match quality**  \n{r.get('_confidence', '')}")

                if "description" in results.columns and str(r.get("description", "")).strip():
                    st.markdown(
                        engine.highlight_text(r["description"], query),
                        unsafe_allow_html=True,
                    )

                if "keywords" in results.columns and str(r.get("keywords", "")).strip():
                    st.caption("Keywords: " + str(r["keywords"]).replace(" | ", ", "))

                if "unit" in results.columns and str(r.get("unit", "")).strip():
                    st.write("Unit:", r["unit"])

                if "spatial_level" in results.columns and str(r.get("spatial_level", "")).strip():
                    st.write("Spatial level:", r["spatial_level"])

                if "time_coverage" in results.columns and str(r.get("time_coverage", "")).strip():
                    st.write("Time coverage:", r["time_coverage"])

                if "access" in results.columns and str(r.get("access", "")).strip():
                    st.write("Access:", r["access"])

                if "_score" in results.columns:
                    st.caption(f"Relevance score: {r['_score']}")

                if "_match_reason" in results.columns and str(r["_match_reason"]).strip():
                    st.caption("Why it matched: " + str(r["_match_reason"]))

                if "_matched_fields" in results.columns and str(r["_matched_fields"]).strip():
                    st.caption("Matched fields: " + str(r["_matched_fields"]))

                related_papers = papers_catalog.related_to_datasets([r["dataset_name"]], limit=2)
                papers_heading = "Related papers"
                if related_papers.empty:
                    related_papers = papers_catalog.search(
                        query=f"{r['dataset_name']} {r['domain']}",
                        limit=2,
                        min_score=60.0,
                    )
                    papers_heading = "Relevant papers"

                if not related_papers.empty:
                    st.markdown(f"**{papers_heading}**")
                    for _, paper in related_papers.iterrows():
                        st.markdown(
                            f"- [{paper['paper_title']}]({paper['paper_link']}) "
                            f"({paper['year']}, {paper['publication']}). Uses: {paper['datasets_used']}."
                        )

                st.markdown(f"[Source]({r['link']})")
                st.divider()
    else:
        catalog_view = engine.catalog(
            domain_filter=domain_filter or None,
            institution_filter=institution_filter or None,
            source_type_filter=source_type_filter or None,
            access_filter=access_filter or None,
        )
        if catalog_view.empty:
            st.warning("No datasets match the current filters.")
        else:
            if domain_filter or institution_filter or source_type_filter or access_filter:
                st.caption(f"{len(catalog_view)} dataset(s) match the current filters")
                st.dataframe(catalog_view.head(result_limit), use_container_width=True, hide_index=True)
            else:
                st.info("Enter a query above or browse the featured catalog below.")
                st.dataframe(engine.featured(limit=result_limit), use_container_width=True, hide_index=True)

            st.download_button(
                "Download current catalog view (CSV)",
                export_csv(catalog_view),
                "dataset_catalog_view.csv",
                mime="text/csv",
            )

elif page == "Research Papers":
    st.header("Research Papers")
    st.caption(
        "Search a curated layer of Portugal-focused empirical papers and see which datasets they use."
    )

    papers_catalog = load_papers_catalog(APP_DATA_VERSION)

    with st.sidebar:
        st.subheader("Paper filters")
        paper_dataset_filter = st.multiselect(
            "Datasets used",
            papers_catalog.dataset_options,
        )
        paper_topic_filter = st.multiselect(
            "Topic",
            papers_catalog.topic_options,
        )
        paper_publication_type_filter = st.multiselect(
            "Publication type",
            papers_catalog.publication_types,
        )
        paper_year_range = st.slider(
            "Year range",
            min_value=papers_catalog.year_min,
            max_value=papers_catalog.year_max,
            value=(papers_catalog.year_min, papers_catalog.year_max),
        )
        paper_include_broad_matches = st.checkbox(
            "Include broader paper matches",
            value=False,
            help="Useful for vague queries. Leave off for cleaner results.",
        )
        paper_result_limit = st.slider("Maximum paper results", min_value=5, max_value=25, value=10)

    with st.expander("Paper search tips", expanded=False):
        st.write(
            "Try queries such as `training`, `gender wage gap`, `credit risk`, "
            "`innovation`, `migration health`, `cpi microdata`, or `input-output`."
        )

    paper_query = st.text_input(
        "Search papers",
        placeholder="e.g. unemployment transitions, gender wage gap, credit risk, innovation",
    )

    selected_year_range = None
    if paper_year_range != (papers_catalog.year_min, papers_catalog.year_max):
        selected_year_range = paper_year_range

    if paper_query:
        paper_suggestions = papers_catalog.suggest(paper_query)
        if paper_suggestions:
            st.caption("Suggestions: " + " | ".join(paper_suggestions))

        paper_results = papers_catalog.search(
            query=paper_query,
            limit=paper_result_limit,
            dataset_filter=paper_dataset_filter or None,
            topic_filter=paper_topic_filter or None,
            publication_type_filter=paper_publication_type_filter or None,
            year_range=selected_year_range,
            include_broad_matches=paper_include_broad_matches,
        )

        if paper_results.empty:
            st.warning("No high-confidence papers found for this query.")
            if not paper_include_broad_matches:
                st.info("Turn on `Include broader paper matches` in the sidebar if you want looser results.")
        else:
            st.caption(f"{len(paper_results)} paper(s) ranked by relevance")
            export_columns = [
                column
                for column in papers_catalog.public_columns + ["_score", "_confidence", "_match_reason"]
                if column in paper_results.columns
            ]
            st.download_button(
                "Download paper results (CSV)",
                export_csv(paper_results[export_columns]),
                "paper_search_results.csv",
                mime="text/csv",
            )

            for _, paper in paper_results.iterrows():
                st.markdown(
                    f"### {papers_catalog.highlight_text(paper['paper_title'], paper_query)}",
                    unsafe_allow_html=True,
                )
                meta_cols = st.columns(4)
                meta_cols[0].markdown(f"**Authors**  \n{paper['authors']}")
                meta_cols[1].markdown(f"**Year**  \n{paper['year']}")
                meta_cols[2].markdown(f"**Publication**  \n{paper['publication']}")
                meta_cols[3].markdown(f"**Match quality**  \n{paper.get('_confidence', '')}")

                st.markdown(f"**Publication type:** {paper['publication_type']}")
                st.markdown(f"**Topic:** {paper['topic']}")
                st.markdown(f"**Methods:** {paper['methods']}")
                st.markdown(f"**Datasets used:** {paper['datasets_used']}")
                st.markdown(f"**Data notes:** {paper['data_notes']}")
                st.markdown(f"**Access:** {paper['access']}")

                if "_score" in paper_results.columns:
                    st.caption(f"Relevance score: {paper['_score']}")
                if "_match_reason" in paper_results.columns and str(paper["_match_reason"]).strip():
                    st.caption("Why it matched: " + str(paper["_match_reason"]))
                if "_matched_fields" in paper_results.columns and str(paper["_matched_fields"]).strip():
                    st.caption("Matched fields: " + str(paper["_matched_fields"]))

                st.markdown(f"[Paper page]({paper['paper_link']})")
                st.divider()
    else:
        paper_catalog_view = papers_catalog.catalog(
            dataset_filter=paper_dataset_filter or None,
            topic_filter=paper_topic_filter or None,
            publication_type_filter=paper_publication_type_filter or None,
            year_range=selected_year_range,
        )
        if paper_catalog_view.empty:
            st.warning("No papers match the current filters.")
        else:
            if paper_dataset_filter or paper_topic_filter or paper_publication_type_filter or selected_year_range:
                st.caption(f"{len(paper_catalog_view)} paper(s) match the current filters")
                st.dataframe(paper_catalog_view.head(paper_result_limit), use_container_width=True, hide_index=True)
            else:
                st.info("Enter a query above or browse the featured paper layer below.")
                st.dataframe(
                    papers_catalog.featured(limit=paper_result_limit),
                    use_container_width=True,
                    hide_index=True,
                )
                st.subheader("Datasets Most Used In The Paper Layer")
                st.dataframe(
                    papers_catalog.dataset_usage_summary(limit=paper_result_limit),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Download current paper view (CSV)",
                export_csv(paper_catalog_view),
                "papers_catalog_view.csv",
                mime="text/csv",
            )

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
