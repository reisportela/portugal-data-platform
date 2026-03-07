import streamlit as st
import pandas as pd
import plotly.express as px
from urllib.parse import urlparse

from modules.papers_catalog import PapersCatalog
from modules.search_engine import SearchEngine
from modules.source_suggestions import REVIEW_STATUSES, SourceSuggestionsStore
from modules.econometrics import run_ols
from modules.observatory import fake_indicator
from modules.export import export_csv


APP_DATA_VERSION = "2026-03-07-2"


def safe_unique_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(value for value in frame[column].astype(str).unique() if value)


def secret_value(name: str, default: str = "") -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return default


def is_valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url).strip())
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


@st.cache_resource(show_spinner=False)
def load_search_engine(cache_version: str):
    return SearchEngine("dataset_catalog.csv")


@st.cache_resource(show_spinner=False)
def load_papers_catalog(cache_version: str):
    return PapersCatalog("data/papers_catalog.csv")


@st.cache_resource(show_spinner=False)
def load_suggestions_store(cache_version: str):
    return SourceSuggestionsStore("data/source_suggestions.sqlite3")


st.set_page_config(page_title="Portugal Data Platform", layout="wide")

st.title("Portugal Data Platform")

admin_review_password = secret_value("suggestions_admin_password")
navigation_options = [
    "Home",
    "Dataset Search",
    "Research Papers",
    "Suggest a Source",
    "Data Lab",
    "Economic Observatory",
]
if admin_review_password:
    navigation_options.insert(4, "Suggestion Inbox")

page = st.sidebar.selectbox("Navigation", navigation_options)

if page == "Home":
    st.write(
        """
Welcome to the Portugal Data Platform.

This portal helps discover Portuguese datasets,
find papers that use them, suggest new sources for review, run simple econometric analyses, and explore economic indicators.
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

elif page == "Suggest a Source":
    st.header("Suggest a Source")
    st.caption(
        "Suggest a Portuguese dataset, portal, API, or documentation source. "
        "Submissions go into a moderation inbox and never appear in public search automatically."
    )

    suggestions_store = load_suggestions_store(APP_DATA_VERSION)

    st.info(
        f"There {'is' if suggestions_store.pending_count() == 1 else 'are'} currently "
        f"{suggestions_store.pending_count()} suggestion(s) waiting for review."
    )

    with st.expander("Submission guidelines", expanded=False):
        st.write(
            "Prioritise official Portuguese data sources, stable documentation pages, clear source ownership, "
            "and concise notes on why the source is useful for teaching or research."
        )

    with st.form("source_suggestion_form", clear_on_submit=True):
        dataset_name = st.text_input("Source or dataset name *")
        institution = st.text_input("Institution or provider *")
        domain = st.text_input("Domain *", placeholder="labour, health, finance, education, municipal")
        link = st.text_input("Source link *", placeholder="https://...")

        form_cols = st.columns(3)
        source_type = form_cols[0].selectbox(
            "Source type",
            [
                "",
                "Portal",
                "API",
                "Dataset",
                "Microdata",
                "Documentation",
                "Survey",
                "Accounts",
                "Indicator set",
                "Proprietary database",
            ],
        )
        access = form_cols[1].selectbox(
            "Access",
            ["", "Open", "Restricted", "Proprietary", "Unknown"],
        )
        spatial_level = form_cols[2].text_input(
            "Spatial level",
            placeholder="national, regional, municipal, firm level",
        )

        description = st.text_area(
            "Why should this be included? *",
            placeholder="Explain what the source contains and why it is useful for Portugal-focused research or teaching.",
        )
        keywords = st.text_area(
            "Keywords",
            placeholder="labour | unemployment | wages | households",
        )
        submission_notes = st.text_area(
            "Additional notes for the curator",
            placeholder="Coverage, licensing, caveats, known papers using it, or access constraints.",
        )

        submitter_cols = st.columns(2)
        submitter_name = submitter_cols[0].text_input("Your name")
        submitter_email = submitter_cols[1].text_input("Your email")

        submitted = st.form_submit_button("Send suggestion for review")

    if submitted:
        errors = []
        if not str(dataset_name).strip():
            errors.append("`Source or dataset name` is required.")
        if not str(institution).strip():
            errors.append("`Institution or provider` is required.")
        if not str(domain).strip():
            errors.append("`Domain` is required.")
        if not str(link).strip():
            errors.append("`Source link` is required.")
        elif not is_valid_http_url(link):
            errors.append("`Source link` must be a valid `http` or `https` URL.")
        if not str(description).strip():
            errors.append("`Why should this be included?` is required.")
        if submitter_email and "@" not in str(submitter_email):
            errors.append("`Your email` must look like an email address.")

        if errors:
            st.error("\n".join(errors))
        else:
            suggestion_id = suggestions_store.submit_suggestion(
                {
                    "dataset_name": dataset_name,
                    "institution": institution,
                    "domain": domain,
                    "link": link,
                    "description": description,
                    "keywords": keywords,
                    "source_type": source_type,
                    "access": access,
                    "spatial_level": spatial_level,
                    "submitter_name": submitter_name,
                    "submitter_email": submitter_email,
                    "submission_notes": submission_notes,
                }
            )
            st.success(
                f"Suggestion #{suggestion_id} was saved for review. "
                "It will not appear in the public search until a curator approves and promotes it."
            )

elif page == "Suggestion Inbox":
    st.header("Suggestion Inbox")
    st.caption(
        "Curator-only review queue for suggested data sources. Approval here does not publish a source automatically."
    )

    if not admin_review_password:
        st.warning("Set `suggestions_admin_password` in Streamlit secrets to enable the curator inbox.")
    else:
        entered_password = st.text_input("Curator password", type="password")
        if entered_password != admin_review_password:
            st.info("Enter the curator password to review source suggestions.")
        else:
            suggestions_store = load_suggestions_store(APP_DATA_VERSION)

            inbox_cols = st.columns(3)
            review_filter = inbox_cols[0].selectbox(
                "Review status",
                ["pending_review", "approved", "rejected", "all"],
            )
            inbox_limit = inbox_cols[1].slider("Rows to display", min_value=5, max_value=100, value=25)
            approved_candidates = suggestions_store.approved_catalog_candidates()
            inbox_cols[2].metric("Approved not yet promoted", len(approved_candidates))

            suggestions_view = suggestions_store.list_suggestions(
                None if review_filter == "all" else review_filter
            )

            if suggestions_view.empty:
                st.warning("No suggestions found for the current filter.")
            else:
                st.dataframe(
                    suggestions_view.head(inbox_limit),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download suggestions (CSV)",
                    export_csv(suggestions_view),
                    "source_suggestions.csv",
                    mime="text/csv",
                )

                selected_id = st.selectbox(
                    "Suggestion to review",
                    suggestions_view["id"].astype(int).tolist(),
                )
                selected_row = suggestions_view[suggestions_view["id"] == selected_id].iloc[0]

                st.markdown(f"### {selected_row['dataset_name']}")
                st.markdown(f"**Institution:** {selected_row['institution']}")
                st.markdown(f"**Domain:** {selected_row['domain']}")
                st.markdown(f"**Link:** {selected_row['link']}")
                st.markdown(f"**Description:** {selected_row['description']}")
                if str(selected_row["keywords"]).strip():
                    st.markdown(f"**Keywords:** {selected_row['keywords']}")
                if str(selected_row["submission_notes"]).strip():
                    st.markdown(f"**Submission notes:** {selected_row['submission_notes']}")
                if str(selected_row["submitter_name"]).strip() or str(selected_row["submitter_email"]).strip():
                    st.markdown(
                        f"**Submitted by:** {selected_row['submitter_name']} {selected_row['submitter_email']}".strip()
                    )
                st.caption(
                    f"Submitted at {selected_row['submitted_at']} | Current status: {selected_row['review_status']}"
                )

                with st.form("review_suggestion_form"):
                    current_status = str(selected_row["review_status"])
                    status_index = REVIEW_STATUSES.index(current_status) if current_status in REVIEW_STATUSES else 0
                    new_status = st.selectbox(
                        "Set review status",
                        REVIEW_STATUSES,
                        index=status_index,
                    )
                    curator_notes = st.text_area(
                        "Curator notes",
                        value=str(selected_row["curator_notes"]),
                        placeholder="Why was this approved or rejected? Any edits needed before promotion?",
                    )
                    review_submitted = st.form_submit_button("Save review decision")

                if review_submitted:
                    suggestions_store.update_review(selected_id, new_status, curator_notes)
                    st.success(
                        "Review status updated. Approved suggestions remain outside public search until you manually add them to the catalog."
                    )

                if not approved_candidates.empty:
                    st.subheader("Approved Catalog Candidates")
                    st.dataframe(
                        approved_candidates,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download approved candidates (CSV)",
                        export_csv(approved_candidates),
                        "approved_source_candidates.csv",
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
