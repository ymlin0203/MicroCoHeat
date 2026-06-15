# app_microcoheat.py
# Usage:
#   streamlit run app_microcoheat.py

import io
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests


# =========================
# Utility functions
# =========================

def _infer_sep(name: str) -> str:
    """Infer file separator from extension."""
    ext = Path(name).suffix.lower()
    return "," if ext == ".csv" else "\t"


def _clean_taxon_prefix(x: str) -> str:
    """Remove common taxonomy prefixes such as g__, s__, D_5__."""
    x = str(x).strip()
    x = re.sub(r"^[a-zA-Z]__", "", x)
    x = re.sub(r"^D_\d+__", "", x)
    return x.strip()


def _last_token(x: str) -> str:
    """
    Extract the last taxonomic rank from a taxonomy string.

    Example:
    Bacteria|Bacillota|Clostridia|Lachnospirales|Lachnospiraceae|Lachnoclostridium|Lachnoclostridium_phytofermentans
    -> Lachnoclostridium_phytofermentans
    """
    x = str(x).strip()
    parts = re.split(r"[|;]", x)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return _clean_taxon_prefix(x)

    return _clean_taxon_prefix(parts[-1])


def _genus_token(x: str) -> str:
    """
    Extract genus from taxonomy string if possible.
    If no explicit genus prefix is found, use the first part before underscore.
    """
    x = str(x).strip()
    parts = re.split(r"[|;]", x)
    parts = [p.strip() for p in parts if p.strip()]

    for p in reversed(parts):
        if p.startswith("g__"):
            return p.replace("g__", "").strip()
        if p.startswith("D_5__"):
            return p.replace("D_5__", "").strip()

    last = _last_token(x)

    if "_" in last:
        return last.split("_")[0].strip()

    return last.strip()


def _species_token(x: str):
    """
    Extract species name from taxonomy string.
    Return None if species-level information is not detected.
    """
    x = str(x).strip()
    parts = re.split(r"[|;]", x)
    parts = [p.strip() for p in parts if p.strip()]

    # Explicit species label
    for p in reversed(parts):
        p = p.strip()

        if p.startswith("s__"):
            sp = p.replace("s__", "").strip()
            if sp and sp.lower() not in ["unassigned", "unknown", "uncultured", "none", "nan"]:
                return sp

        if p.startswith("D_6__"):
            sp = p.replace("D_6__", "").strip()
            if sp and sp.lower() not in ["unassigned", "unknown", "uncultured", "none", "nan"]:
                return sp

    # Fallback: last rank looks like Genus_species
    last = _last_token(x)

    if "_" in last:
        pieces = last.split("_")
        if len(pieces) >= 2 and pieces[0] and pieces[1]:
            if last.lower() not in ["unassigned", "unknown", "uncultured", "none", "nan"]:
                return last

    return None


def format_display_label(x: str, mode: str = "Last taxonomic rank") -> str:
    """
    Format labels for heatmap display only.
    This does not change the actual analysis table or downloaded CSV.
    """
    x = str(x).strip()

    if mode == "Original":
        return x

    if mode == "Last taxonomic rank":
        return _last_token(x)

    if mode == "Species only":
        sp = _species_token(x)
        if sp is not None:
            return sp
        return _last_token(x)

    return x


def shorten_label(x: str, max_len: int = 35) -> str:
    """Shorten long labels for plotting only."""
    x = str(x)
    if len(x) <= max_len:
        return x
    return x[:max_len] + "..."


@st.cache_data(show_spinner=False)
def read_table(uploaded_file) -> pd.DataFrame:
    """Read uploaded abundance table."""
    sep = _infer_sep(getattr(uploaded_file, "name", ""))
    df = pd.read_csv(uploaded_file, sep=sep, index_col=0)
    return df


def parse_manual_taxa(taxa_input: str) -> List[str]:
    """
    Parse manual taxa input.
    Supports one-per-line or comma-separated input.
    """
    taxa_list = [
        x.strip()
        for part in str(taxa_input).split("\n")
        for x in part.split(",")
        if x.strip()
    ]
    return taxa_list


def prepare_taxa_table(
    df: pd.DataFrame,
    taxa_list: List[str],
    label_mode: str = "Use species-level only",
    match_mode: str = "Exact match",
    case_sensitive: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare taxa/features x samples table.

    Rule:
    - If taxa_list is empty, use all taxa/features after selected label mode.
    - If taxa_list is not empty, filter by user input.
    """
    df2 = df.copy()

    # Convert all values to numeric
    df2 = (
        df2.astype(str)
        .replace("%", "", regex=True)
        .replace(",", "", regex=True)
        .replace("-", "0")
        .replace("NA", "0")
        .replace("N/A", "0")
        .replace("nan", "0")
        .replace("None", "0")
    )
    df2 = df2.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Handle row labels
    if label_mode == "Use table labels as-is":
        df2.index = pd.Index(df2.index.astype(str))

    elif label_mode == "Use last taxonomic rank":
        df2.index = pd.Index([_last_token(i) for i in df2.index])
        df2 = df2.groupby(df2.index).sum()

    elif label_mode == "Use species-level only":
        species_names = [_species_token(i) for i in df2.index]
        keep = [s is not None for s in species_names]

        df2 = df2.loc[keep]
        species_names = [s for s in species_names if s is not None]

        df2.index = pd.Index(species_names)
        df2 = df2.groupby(df2.index).sum()

    elif label_mode == "Use genus-level and merge":
        df2.index = pd.Index([_genus_token(i) for i in df2.index])
        df2 = df2.groupby(df2.index).sum()

    else:
        raise ValueError(f"Unknown label mode: {label_mode}")

    # Clean blank labels
    df2.index = pd.Index([str(i).strip() for i in df2.index])
    df2 = df2.loc[df2.index != ""]
    df2 = df2.loc[
        ~df2.index.str.lower().isin(
            ["nan", "none", "unassigned", "unknown", "uncultured"]
        )
    ]

    missing_terms: List[str] = []

    # Manual filtering
    if taxa_list:
        index_values = df2.index.astype(str)

        if case_sensitive:
            index_for_match = index_values
            taxa_for_match = taxa_list
        else:
            index_for_match = pd.Index([i.lower() for i in index_values])
            taxa_for_match = [t.lower() for t in taxa_list]

        keep_mask = np.zeros(len(df2), dtype=bool)

        if match_mode == "Exact match":
            taxa_set = set(taxa_for_match)
            keep_mask = np.array([x in taxa_set for x in index_for_match])

            matched_terms = set(index_for_match[keep_mask])
            missing_terms = [
                original
                for original, query in zip(taxa_list, taxa_for_match)
                if query not in matched_terms
            ]

        elif match_mode == "Contains match":
            matched_any = []

            for original, query in zip(taxa_list, taxa_for_match):
                current_mask = np.array([query in x for x in index_for_match])
                keep_mask = keep_mask | current_mask
                matched_any.append(bool(current_mask.any()))

            missing_terms = [
                original
                for original, ok in zip(taxa_list, matched_any)
                if not ok
            ]

        else:
            raise ValueError(f"Unknown match mode: {match_mode}")

        df2 = df2.loc[keep_mask]

    # Remove all-zero taxa
    if not df2.empty:
        df2 = df2.loc[df2.sum(axis=1) > 0]

    # Remove taxa with no variation across samples
    if not df2.empty:
        df2 = df2.loc[df2.var(axis=1) > 0]

    return df2, missing_terms


# =========================
# Statistics
# =========================

@st.cache_data(show_spinner=False)
def spearman_corr_and_p(
    df: pd.DataFrame,
    fdr_alpha: float,
    method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate pairwise Spearman correlation among rows.

    FDR correction:
    - Only upper triangle without diagonal is corrected.
    - Corrected p-values are mirrored back to the full matrix.
    """
    taxa = df.index.astype(str)
    n = len(taxa)

    corr = np.eye(n, dtype=float)
    p_raw = np.zeros((n, n), dtype=float)

    values = df.to_numpy(dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            r, p = spearmanr(values[i, :], values[j, :])

            if np.isnan(r):
                r = 0.0
            if np.isnan(p):
                p = 1.0

            corr[i, j] = corr[j, i] = r
            p_raw[i, j] = p_raw[j, i] = p

    p_corr = np.ones((n, n), dtype=float)
    np.fill_diagonal(p_corr, 0.0)

    if n > 1:
        iu = np.triu_indices(n, k=1)
        pvals = p_raw[iu]

        _, pvals_corr, _, _ = multipletests(
            pvals,
            alpha=fdr_alpha,
            method=method,
            is_sorted=False,
            returnsorted=False,
        )

        p_corr[iu] = pvals_corr
        p_corr[(iu[1], iu[0])] = pvals_corr

    corr_df = pd.DataFrame(corr, index=taxa, columns=taxa)
    p_df = pd.DataFrame(p_corr, index=taxa, columns=taxa)

    return corr_df, p_df


@st.cache_data(show_spinner=False)
def reorder_by_clustering(
    corr_df: pd.DataFrame,
    p_df: pd.DataFrame,
    method: str = "average",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reorder correlation matrix by hierarchical clustering.

    Clustering uses the full correlation matrix,
    not the FDR-masked matrix.
    """
    if corr_df.shape[0] <= 2:
        return corr_df, p_df

    corr = corr_df.copy().astype(float)

    # Force symmetry
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr.values, 1.0)

    # Correlation distance
    dist = 1 - corr
    dist = dist.clip(lower=0)
    np.fill_diagonal(dist.values, 0.0)

    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)

    return corr_df.iloc[order, order], p_df.iloc[order, order]


# =========================
# Plot function
# =========================

def draw_heatmap(
    corr_df_ord: pd.DataFrame,
    p_df_ord: pd.DataFrame,
    fdr_alpha: float,
    show_mode: str,
    cmap: str,
    fig_w_cm: float,
    fig_h_cm: float,
    font_size: int,
    dpi: int,
    linewidths: float,
    shorten_plot_labels: bool,
    max_label_len: int,
    display_label_mode: str,
) -> plt.Figure:
    """Draw heatmap."""
    if show_mode == "Show significant only":
        plot_values = corr_df_ord.where(p_df_ord <= fdr_alpha)
    else:
        plot_values = corr_df_ord.copy()

    plot_df = plot_values.copy()

    # Display label formatting only
    plot_df.index = [
        format_display_label(i, mode=display_label_mode)
        for i in plot_df.index
    ]
    plot_df.columns = [
        format_display_label(i, mode=display_label_mode)
        for i in plot_df.columns
    ]

    # Optional shortening after extracting last rank/species
    if shorten_plot_labels:
        plot_df.index = [
            shorten_label(i, max_len=max_label_len)
            for i in plot_df.index
        ]
        plot_df.columns = [
            shorten_label(i, max_len=max_label_len)
            for i in plot_df.columns
        ]

    matplotlib.rc("font", size=font_size)

    fig, ax = plt.subplots(
        figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54),
        dpi=dpi,
        constrained_layout=True,
    )

    sns.heatmap(
        plot_df,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        xticklabels=True,
        yticklabels=True,
        annot=False,
        linewidths=linewidths,
        linecolor="white",
        cbar_kws={
            "label": "Spearman correlation coefficient",
            "shrink": 0.75,
        },
        ax=ax,
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
        va="top",
        fontsize=font_size,
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=font_size,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(
        "Spearman correlation coefficient",
        fontsize=font_size,
    )

    return fig


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="MicroCoHeat — Microbial Co-occurrence Heatmap",
    layout="wide",
)

st.title("🧬 MicroCoHeat — Microbial Co-occurrence Heatmap")
st.caption(
    "Upload a taxa/species/genus/ASV abundance table. "
    "Rows = taxa/features, columns = samples."
)

with st.sidebar:
    st.header("📄 Upload")

    uploaded = st.file_uploader(
        "Upload table (.tsv/.txt/.csv)",
        type=["tsv", "txt", "csv"],
        help=(
            "The first column should be taxon/species/genus/feature ID. "
            "Other columns should be samples."
        ),
    )

    transpose_table = st.checkbox(
        "Transpose table",
        value=False,
        help=(
            "Turn samples × taxa into taxa × samples. "
            "Use this if your bacteria are columns."
        ),
    )

    st.header("🔎 Taxa / Species filter")

    label_mode = st.selectbox(
        "Taxa label mode",
        [
            "Use table labels as-is",
            "Use last taxonomic rank",
            "Use species-level only",
            "Use genus-level and merge",
        ],
        index=2,
        help=(
            "Use species-level only: keep species-level rows only. "
            "Example: Bacteria|...|Lachnoclostridium_phytofermentans "
            "will become Lachnoclostridium_phytofermentans."
        ),
    )

    taxa_input = st.text_area(
        "Enter bacteria names",
        value="",
        height=150,
        placeholder=(
            "Example:\n"
            "Lachnoclostridium_phytofermentans\n"
            "Streptococcus_gordonii\n"
            "Fusobacterium_nucleatum"
        ),
        help=(
            "Optional. Enter one per line or comma-separated. "
            "Leave empty to use all taxa/features under the selected label mode."
        ),
    )

    match_mode = st.radio(
        "Manual filter matching",
        ["Exact match", "Contains match"],
        index=0,
        help=(
            "Exact match is stricter. "
            "Contains match is useful when typing partial names, "
            "for example Streptococcus."
        ),
    )

    case_sensitive = st.checkbox(
        "Case-sensitive matching",
        value=False,
    )

    st.header("📐 Statistics")

    fdr_alpha = st.number_input(
        "FDR α",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
    )

    p_adjust_method = st.selectbox(
        "P-value correction",
        [
            "fdr_bh",
            "bonferroni",
            "holm",
            "fdr_by",
            "sidak",
            "holm-sidak",
        ],
        index=0,
    )

    cluster_method = st.selectbox(
        "Clustering method",
        [
            "average",
            "complete",
            "single",
            "weighted",
        ],
        index=0,
        help="Average linkage is recommended for correlation-distance heatmaps.",
    )

    st.header("🎨 Plot")

    display_label_mode = st.selectbox(
        "Heatmap label display",
        [
            "Original",
            "Last taxonomic rank",
            "Species only",
        ],
        index=1,
        help=(
            "Original = show full taxonomy string. "
            "Last taxonomic rank = show only the last part after | or ;. "
            "Species only = try to show species name only."
        ),
    )

    show_mode = st.radio(
        "Heatmap display",
        [
            "Show significant only",
            "Show all correlations",
        ],
        index=0,
        help=(
            "Show significant only will display non-significant cells as blank."
        ),
    )

    cmap = st.selectbox(
        "Colormap",
        [
            "bwr_r",
            "coolwarm",
            "vlag",
            "icefire",
            "RdBu_r",
            "viridis",
        ],
        index=0,
    )

    auto_fig_size = st.checkbox(
        "Auto figure size by taxa number",
        value=True,
    )

    manual_fig_w = st.number_input(
        "Manual width (cm)",
        min_value=8.0,
        max_value=120.0,
        value=30.0,
        step=1.0,
    )

    manual_fig_h = st.number_input(
        "Manual height (cm)",
        min_value=8.0,
        max_value=120.0,
        value=26.0,
        step=1.0,
    )

    font_size = st.number_input(
        "Font size",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
    )

    dpi = st.number_input(
        "DPI",
        min_value=72,
        max_value=600,
        value=300,
        step=10,
    )

    linewidths = st.number_input(
        "Grid line width",
        min_value=0.0,
        max_value=5.0,
        value=0.3,
        step=0.1,
    )

    shorten_plot_labels = st.checkbox(
        "Shorten long labels on heatmap",
        value=False,
    )

    max_label_len = st.number_input(
        "Max label length",
        min_value=10,
        max_value=120,
        value=35,
        step=5,
    )


# =========================
# Main analysis
# =========================

if uploaded is None:
    st.info("Upload a table from the sidebar to start.")
    st.stop()

df_raw = read_table(uploaded)

if transpose_table:
    df_raw = df_raw.T

st.success(
    f"Loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} samples"
)

with st.expander("🔍 Raw uploaded table preview", expanded=False):
    st.dataframe(
        df_raw.iloc[:10, :10],
        use_container_width=True,
    )

taxa_list = parse_manual_taxa(taxa_input)

df, missing_terms = prepare_taxa_table(
    df_raw,
    taxa_list=taxa_list,
    label_mode=label_mode,
    match_mode=match_mode,
    case_sensitive=case_sensitive,
)

if taxa_list:
    st.info(
        f"Manual filter applied. Selected {df.shape[0]} taxa/features."
    )
else:
    st.info(
        f"No manual filter applied. "
        f"Using all available taxa/features under selected mode: {df.shape[0]}."
    )

if missing_terms:
    with st.expander("⚠️ Manual input not found", expanded=True):
        st.write("These terms were not matched:")
        st.dataframe(
            pd.DataFrame({"Not found": missing_terms}),
            use_container_width=True,
        )

if df.empty:
    st.warning(
        "No taxa/features available after filtering. "
        "Please check bacteria names, label mode, or table format."
    )
    st.stop()

if df.shape[0] < 2:
    st.warning(
        "At least two taxa/features are required for correlation analysis."
    )
    st.stop()

if df.shape[1] < 3:
    st.warning(
        "At least three samples are recommended for Spearman correlation."
    )
    st.stop()


# Auto figure size
n_taxa = df.shape[0]

if auto_fig_size:
    fig_w = max(18.0, min(120.0, n_taxa * 0.75 + 10))
    fig_h = max(16.0, min(120.0, n_taxa * 0.55 + 8))
else:
    fig_w = manual_fig_w
    fig_h = manual_fig_h


# Preview tables
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🧾 Taxa/features used for analysis")
    st.dataframe(
        pd.DataFrame({"Taxa / Feature": df.index}),
        use_container_width=True,
        height=360,
    )

with right_col:
    st.subheader("📋 Processed abundance table preview")
    st.dataframe(
        df.iloc[:30, :10],
        use_container_width=True,
        height=360,
    )


# Correlation analysis
with st.spinner("Calculating Spearman correlations..."):
    corr_df, p_df = spearman_corr_and_p(
        df,
        fdr_alpha=fdr_alpha,
        method=p_adjust_method,
    )

    corr_df_ord, p_df_ord = reorder_by_clustering(
        corr_df,
        p_df,
        method=cluster_method,
    )


# Heatmap
st.subheader("🔥 Co-occurrence heatmap")

fig = draw_heatmap(
    corr_df_ord=corr_df_ord,
    p_df_ord=p_df_ord,
    fdr_alpha=fdr_alpha,
    show_mode=show_mode,
    cmap=cmap,
    fig_w_cm=fig_w,
    fig_h_cm=fig_h,
    font_size=font_size,
    dpi=dpi,
    linewidths=linewidths,
    shorten_plot_labels=shorten_plot_labels,
    max_label_len=max_label_len,
    display_label_mode=display_label_mode,
)

st.pyplot(
    fig,
    clear_figure=False,
    use_container_width=True,
)

st.caption(
    f"Figure size: {fig_w:.1f} cm × {fig_h:.1f} cm | "
    f"Taxa/features: {df.shape[0]} | "
    f"Samples: {df.shape[1]} | "
    f"Display mode: {show_mode} | "
    f"Label display: {display_label_mode}"
)


# Result matrices
with st.expander("📊 Correlation matrix, clustered order", expanded=False):
    st.dataframe(
        corr_df_ord,
        use_container_width=True,
    )

with st.expander("📊 Adjusted P-value matrix, same order", expanded=False):
    st.dataframe(
        p_df_ord,
        use_container_width=True,
    )


# =========================
# Downloads
# =========================

st.subheader("⬇️ Download results")

csv_corr = corr_df_ord.to_csv().encode("utf-8-sig")
csv_p = p_df_ord.to_csv().encode("utf-8-sig")
csv_used = df.to_csv().encode("utf-8-sig")

st.download_button(
    "Download correlation CSV",
    data=csv_corr,
    file_name="microcoheat_corr_clustered.csv",
    mime="text/csv",
)

st.download_button(
    "Download adjusted p-values CSV",
    data=csv_p,
    file_name="microcoheat_pvalues_clustered.csv",
    mime="text/csv",
)

st.download_button(
    "Download processed taxa table CSV",
    data=csv_used,
    file_name="microcoheat_processed_taxa_table.csv",
    mime="text/csv",
)

buf_png = io.BytesIO()
buf_pdf = io.BytesIO()

fig.savefig(
    buf_png,
    format="png",
    dpi=dpi,
    bbox_inches="tight",
)

fig.savefig(
    buf_pdf,
    format="pdf",
    bbox_inches="tight",
)

buf_png.seek(0)
buf_pdf.seek(0)

st.download_button(
    "Download heatmap PNG",
    data=buf_png,
    file_name="microcoheat_heatmap.png",
    mime="image/png",
)

st.download_button(
    "Download heatmap PDF",
    data=buf_pdf,
    file_name="microcoheat_heatmap.pdf",
    mime="application/pdf",
)

plt.close(fig)
