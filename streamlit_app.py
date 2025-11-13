# streamlit_app.py

from pathlib import Path
import ntpath

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).parent


def _filename_from_path(path_str: str) -> str:
    """
    Take whatever is in f'file' (full Windows path) and return just the PDF filename.
    Works even if the path string has extra quotes.
    """
    if pd.isna(path_str):
        return ""
    s = str(path_str).strip("'\"")
    return ntpath.basename(s)


def _explode_top_labels(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Take the merged advisory + fraud-mapping dataframe and turn the
    'top_labels_regex' column into one row per (advisory, fraud_type)
    with a 'count' column.
    Example value: "money_laundering:4; structuring_smurfing:1"
    """
    records = []

    for _, row in merged.iterrows():
        # ← use the real column name from fincen_ocr_fraud_mapper.py
        labels_raw = row.get("top_labels_regex", "")
        if pd.isna(labels_raw):
            continue

        labels_str = str(labels_raw).strip()
        if not labels_str:
            continue

        parts = [p.strip() for p in labels_str.split(";") if p.strip()]
        for part in parts:
            if ":" not in part:
                fraud_type = part.strip()
                count = 1
            else:
                fraud_type, count_str = part.split(":", 1)
                fraud_type = fraud_type.strip()
                try:
                    count = int(count_str.strip())
                except ValueError:
                    count = 1

            records.append(
                {
                    "pdf_filename": row.get("pdf_filename", ""),
                    "title": row.get("title", ""),
                    "year": row.get("year", None),
                    "fraud_type": fraud_type,
                    # ← use total_hits_regex here
                    "count": count,
                    "total_hits_file": row.get("total_hits_regex", 0),
                    "pdf_url": row.get("pdf_url", ""),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "pdf_filename",
                "title",
                "year",
                "fraud_type",
                "count",
                "total_hits_file",
                "pdf_url",
            ]
        )

    return pd.DataFrame(records)



@st.cache_data
def load_data():
    """Load and join advisory metadata with fraud-mapping summary."""
    adv_path = DATA_DIR / "fincen_advisories.csv"
    mapping_path = DATA_DIR / "fincen_fraud_mapping.csv"

    advisories = pd.read_csv(adv_path)
    mapping = pd.read_csv(mapping_path)

    # Parse advisory dates and derive year
    advisories["date"] = pd.to_datetime(advisories["date"])
    advisories["year"] = advisories["date"].dt.year

    # Normalize the path in 'file' and extract the local pdf filename
    mapping["pdf_basename"] = mapping["file"].apply(_filename_from_path)

    # Join mapping -> advisory metadata via local filename
    merged = mapping.merge(
        advisories,
        left_on="pdf_basename",
        right_on="pdf_filename",
        how="left",
        suffixes=("_map", ""),
    )

    # Explode top_labels into one row per fraud type
    exploded = _explode_top_labels(merged)

    return advisories, mapping, merged, exploded


def main():
    st.set_page_config(page_title="FinCEN Fraud Advisory Explorer", layout="wide")
    st.title("🔍 FinCEN Fraud Advisory Explorer")

    st.markdown(
        """
        This app lets you explore **FinCEN advisories** and how often different
        fraud types appear over time, based on the outputs of your
        `fincen_ocr_fraud_mapper.py`.

        Use the filters in the sidebar to zoom in on particular years or fraud types.
        """
    )

    advisories, mapping, merged, exploded = load_data()

    if exploded.empty:
        st.warning(
            "No fraud labels found in `fincen_fraud_mapping.csv`. "
            "Make sure `top_labels_regex` is populated by `fincen_ocr_fraud_mapper.py`."
        )
        return

    # ---------- Sidebar filters ----------
    with st.sidebar:
        st.header("Filters")

        min_year = int(exploded["year"].min())
        max_year = int(exploded["year"].max())
        year_range = st.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
        )

        all_fraud_types = sorted(exploded["fraud_type"].dropna().unique().tolist())
        selected_types = st.multiselect(
            "Fraud types",
            options=all_fraud_types,
            default=all_fraud_types,
        )

    # Apply filters
    mask = exploded["year"].between(year_range[0], year_range[1])
    if selected_types:
        mask &= exploded["fraud_type"].isin(selected_types)

    filtered = exploded[mask].copy()

    if filtered.empty:
        st.warning("No advisories match the current filters.")
        return

    # ---------- KPIs ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("Advisories", filtered["pdf_filename"].nunique())
    col2.metric("Total keyword hits", int(filtered["count"].sum()))
    col3.metric("Fraud types", filtered["fraud_type"].nunique())

    st.markdown("---")

    # ---------- Timeline chart ----------
    st.subheader("Timeline of fraud-type keyword hits")

    yearly = (
        filtered.groupby(["year", "fraud_type"])["count"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    # Pivot for Streamlit's built-in chart
    chart_data = yearly.pivot(index="year", columns="fraud_type", values="count").fillna(
        0
    )
    st.line_chart(chart_data)

    # ---------- Advisory table ----------
    st.subheader("Advisories matching filters")

    advis_summary = (
        filtered.groupby(["year", "title", "pdf_filename", "pdf_url"])
        .agg(
            total_hits=("count", "sum"),
            distinct_fraud_types=("fraud_type", "nunique"),
        )
        .reset_index()
        .sort_values(["year", "total_hits"], ascending=[False, False])
    )

    # Show as a table; PDF URL will be clickable in most Streamlit frontends
    st.dataframe(
        advis_summary[
            ["year", "title", "total_hits", "distinct_fraud_types", "pdf_url"]
        ],
        use_container_width=True,
    )

    st.markdown(
        """
        *Tip:* You can sort the table by `total_hits` to see which advisories
        have the strongest signal for the selected fraud types.
        """
    )


if __name__ == "__main__":
    main()
