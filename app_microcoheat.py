# app_microcoheat.py
# Usage:
#   streamlit run app_microcoheat.py
#
# ---------------------------------------------------------------------------
# Performance notes:
#   1. spearman_corr_and_p uses scipy's vectorized matrix form of
#      spearmanr() instead of an O(n^2) Python for-loop calling spearmanr()
#      on every taxon pair individually (100-200x faster on 50-200 taxa).
#   2. prepare_taxa_table / normalize_table are cached (@st.cache_data).
#   3. draw_heatmap builds the on-screen figure at a capped "preview" DPI
#      and only renders at the user's chosen (possibly much higher) DPI at
#      export time in the PNG/PDF savefig calls.
#   4. sns.heatmap(..., rasterized=True) keeps PDF export fast/small.
#   5. Font-size is applied via a scoped plt.rc_context(...) instead of the
#      global matplotlib.rc(...), so it can't leak across sessions.
#   6. All sidebar controls sit inside an st.form(), so adjusting several
#      settings only re-runs the pipeline once, on "Run analysis".
#   7. statsmodels, plotly and networkx are imported lazily, only where
#      used, so `streamlit run` reaches the upload screen faster and the
#      default (network graph / interactive heatmap / group comparison all
#      off) path pays none of their import cost.
#
# Feature notes (new):
#   - Optional data preprocessing before correlation: relative abundance
#     (TSS) or CLR (centered log-ratio), addressing the "compositional
#     data" caveat of running Spearman directly on raw counts.
#   - Optional interactive Plotly heatmap alongside the static one.
#   - Optional co-occurrence network graph (nodes = taxa, edges =
#     significant correlations), with an exportable edge table
#     (Cytoscape/Gephi-ready) and a hub-taxa (degree) summary.
#   - Optional group comparison: upload a metadata table, get one heatmap
#     per group (aligned to the same taxon order) plus a table of taxa
#     pairs whose significance differs between two chosen groups.
# ---------------------------------------------------------------------------

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
# statsmodels, plotly and networkx are imported lazily inside the functions
# that need them (see the perf notes above) rather than at module scope.


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
            if sp and sp.lower() not in [
                "unassigned",
                "unknown",
                "uncultured",
                "none",
                "nan",
            ]:
                return sp

        if p.startswith("D_6__"):
            sp = p.replace("D_6__", "").strip()
            if sp and sp.lower() not in [
                "unassigned",
                "unknown",
                "uncultured",
                "none",
                "nan",
            ]:
                return sp

    # Fallback: last rank looks like Genus_species
    last = _last_token(x)

    if "_" in last:
        pieces = last.split("_")
        if len(pieces) >= 2 and pieces[0] and pieces[1]:
            if last.lower() not in [
                "unassigned",
                "unknown",
                "uncultured",
                "none",
                "nan",
            ]:
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


_MISSING_PLACEHOLDERS = {"-", "NA", "N/A", "nan", "None"}


@st.cache_data(show_spinner=False)
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

    Cached: same uploaded table + same sidebar settings -> no recomputation.
    """
    df2 = df.copy()

    # Convert all values to numeric.
    # Only string-typed (object) columns need cleaning; columns pandas
    # already parsed as numeric skip straight to pd.to_numeric below,
    # which is noticeably faster than round-tripping every cell through
    # str() + several whole-table regex passes on wide/long tables.
    for col in df2.columns:
        if df2[col].dtype == object:
            s = df2[col].astype(str)
            s = s.str.replace("%", "", regex=False)
            s = s.str.replace(",", "", regex=False)
            s = s.where(~s.isin(_MISSING_PLACEHOLDERS), "0")
            df2[col] = s

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
            # dtype=bool is explicit here (not left to be inferred) because
            # an empty df2 (0 taxa left after cleaning) makes this a list
            # comprehension over zero elements, and np.array([]) defaults to
            # float64 -- which then fails the `keep_mask | current_mask`
            # bitwise-or below under numpy>=2's stricter ufunc casting rules
            # (numpy<2 silently allowed it). This affects both match modes.
            keep_mask = np.array(
                [x in taxa_set for x in index_for_match], dtype=bool
            )

            matched_terms = set(index_for_match[keep_mask])
            missing_terms = [
                original
                for original, query in zip(taxa_list, taxa_for_match)
                if query not in matched_terms
            ]

        elif match_mode == "Contains match":
            matched_any = []

            for original, query in zip(taxa_list, taxa_for_match):
                current_mask = np.array(
                    [query in x for x in index_for_match], dtype=bool
                )
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
# Preprocessing / normalization
# =========================

NORM_RAW = "不轉換 (Raw values)"
NORM_TSS = "相對豐度 (TSS, 每個樣本總和為 1)"
NORM_CLR = "CLR (centered log-ratio)"
NORMALIZATION_METHODS = [NORM_RAW, NORM_TSS, NORM_CLR]


@st.cache_data(show_spinner=False)
def normalize_table(df: pd.DataFrame, method: str, pseudocount: float) -> pd.DataFrame:
    """
    Optional preprocessing applied before the correlation step.

    Spearman correlation computed directly on raw counts/relative-abundance
    values can be sensitive to the "compositional" nature of microbiome
    data (all taxa in a sample necessarily sum to a fixed total, which
    induces spurious correlations -- Aitchison 1986; Gloor et al. 2017,
    "Microbiome Datasets Are Compositional").

    IMPORTANT for methods write-ups: TSS only corrects for sequencing-depth
    differences (dividing by each sample's own total) -- it does NOT
    resolve the compositional "constant sum" artifact itself, and because
    the divisor differs sample to sample, TSS *can* change a given taxon's
    rank order across samples too, same as CLR (it is not a pure, rank-
    preserving rescaling relative to raw counts). CLR (centered log-ratio)
    is the transform from the compositional-data literature that actually
    addresses the constant-sum artifact, by replacing raw abundances with
    log-ratios to each sample's own geometric mean. If this analysis is
    going into a publication, CLR is the defensible default to report, not
    TSS or raw counts, and the normalization method used should be stated
    explicitly in the methods section.
    """
    if method == NORM_RAW:
        return df

    if method == NORM_TSS:
        col_sums = df.sum(axis=0).replace(0, np.nan)
        out = df.div(col_sums, axis=1).fillna(0.0)
        return out

    if method == NORM_CLR:
        x = df.to_numpy(dtype=float) + pseudocount
        x = np.clip(x, a_min=1e-12, a_max=None)
        log_x = np.log(x)
        gm = log_x.mean(axis=0, keepdims=True)
        clr = log_x - gm
        return pd.DataFrame(clr, index=df.index, columns=df.columns)

    raise ValueError(f"Unknown normalization method: {method}")


def suggest_pseudocount(df: pd.DataFrame) -> float:
    """Half of the smallest strictly-positive value in the table, as a
    reasonable default CLR pseudocount; falls back to a tiny constant if
    the table has no positive values at all."""
    values = df.to_numpy(dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        return 1e-6
    return float(positive.min() / 2)


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

    Uses scipy's vectorized matrix form of spearmanr() (ranks every row
    once, computes the full correlation/p-value matrices in compiled
    code) instead of looping over every (i, j) pair in Python.

    FDR correction:
    - Only upper triangle without diagonal is corrected (the diagonal is
      not a real test, and correcting it too would understate the
      significance of every real pair).
    - Corrected p-values are mirrored back to the full matrix.

    CAVEAT for methods write-ups: a Spearman correlation is undefined
    (scipy returns NaN) whenever one of the two taxa has zero variance
    across the samples being tested -- this is checked and filtered out
    for the *overall* table (see prepare_taxa_table's variance filter),
    but it can still happen inside a single group's subset in the group-
    comparison feature (e.g. a taxon that is invariant within "Healthy"
    even though it varies overall). Those NaN pairs are silently set to
    r=0, adj_p=1 (i.e. treated as "not significant") rather than left
    undefined/excluded, so they don't crash the heatmap or downstream FDR
    correction. If reporting group-specific results, it's worth checking
    each group's data for taxa with zero within-group variance and noting
    how they were handled, rather than assuming every r=0 reflects a
    genuinely tested null result.
    """
    from statsmodels.stats.multitest import multipletests

    taxa = df.index.astype(str)
    n = len(taxa)

    values = df.to_numpy(dtype=float)

    if n < 2:
        corr = np.eye(n, dtype=float)
        p_raw = np.zeros((n, n), dtype=float)
    else:
        result = spearmanr(values, axis=1)

        # scipy's result object exposes the correlation matrix as either
        # `.correlation` (older scipy) or `.statistic` (newer scipy) --
        # support both so this doesn't silently break on a scipy upgrade.
        corr_raw = getattr(result, "correlation", None)
        if corr_raw is None:
            corr_raw = result.statistic

        corr = np.atleast_2d(np.asarray(corr_raw, dtype=float))
        p_raw = np.atleast_2d(np.asarray(result.pvalue, dtype=float))

        # With exactly 2 taxa, scipy returns scalars rather than a 2x2
        # matrix -- normalize that case so the rest of the code can assume
        # an (n, n) shape.
        if corr.shape != (n, n):
            r = float(corr.flat[0])
            p = float(p_raw.flat[0])
            corr = np.array([[1.0, r], [r, 1.0]])
            p_raw = np.array([[0.0, p], [p, 0.0]])

        corr = np.nan_to_num(corr, nan=0.0)
        p_raw = np.nan_to_num(p_raw, nan=1.0)
        np.fill_diagonal(corr, 1.0)
        np.fill_diagonal(p_raw, 0.0)

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

    # Work on a plain, guaranteed-writable numpy array rather than mutating
    # a DataFrame's `.values` in place: under pandas' Copy-on-Write (the
    # default since pandas 3.0, and optional earlier), `.values` can hand
    # back a read-only view, and np.fill_diagonal(df.values, ...) then
    # raises "underlying array is read-only". np.array(..., copy=True)
    # sidesteps that regardless of the pandas version in use.
    corr = np.array(corr_df, dtype=float, copy=True)

    # Force symmetry
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    # Correlation distance
    dist = 1 - corr
    dist = np.clip(dist, a_min=0.0, a_max=None)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)

    return corr_df.iloc[order, order], p_df.iloc[order, order]


# =========================
# Co-occurrence network helpers
# =========================

@st.cache_data(show_spinner=False)
def build_edge_table(
    corr_df: pd.DataFrame,
    p_df: pd.DataFrame,
    fdr_alpha: float,
) -> pd.DataFrame:
    """
    Turn the correlation/p-value matrices into a "significant pairs only"
    edge list (taxon_1, taxon_2, r, adj_p), sorted by |r| descending.
    This is the same significance test used to mask the heatmap, just
    reshaped into a Cytoscape/Gephi-friendly table.
    """
    taxa = np.array(corr_df.index.astype(str))
    n = len(taxa)

    if n < 2:
        return pd.DataFrame(columns=["taxon_1", "taxon_2", "r", "adj_p"])

    corr = corr_df.to_numpy(dtype=float)
    p = p_df.to_numpy(dtype=float)

    iu = np.triu_indices(n, k=1)
    r_vals = corr[iu]
    p_vals = p[iu]
    mask = p_vals <= fdr_alpha

    edge_df = pd.DataFrame(
        {
            "taxon_1": taxa[iu[0][mask]],
            "taxon_2": taxa[iu[1][mask]],
            "r": r_vals[mask],
            "adj_p": p_vals[mask],
        }
    )

    if edge_df.empty:
        return edge_df

    edge_df = edge_df.reindex(
        edge_df["r"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return edge_df


def draw_network_figure(
    edge_df: pd.DataFrame,
    all_taxa: List[str],
    abundance: Optional[Dict[str, float]],
    layout: str,
    hide_isolated: bool,
    node_size_basis: str,
):
    """Build an interactive Plotly co-occurrence network from the edge
    table. Returns (figure, hub_taxa_dataframe), or (None, None) if there
    is nothing left to draw (e.g. everything filtered out as isolated)."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(all_taxa)
    for _, row in edge_df.iterrows():
        # Deliberately NOT stored under the key "weight": networkx's
        # layout algorithms read a "weight" edge attribute automatically,
        # but disagree on what it means (kamada_kawai treats it as a
        # shortest-path *distance*, spring_layout as an attraction
        # *strength* -- opposite senses), and either way r can be
        # negative, which breaks kamada_kawai's Dijkstra-based layout
        # ("Contradictory paths found: negative weights?"). So layouts
        # below are computed unweighted (topology only); "r" is kept
        # purely for edge coloring/hover text.
        graph.add_edge(row["taxon_1"], row["taxon_2"], r=float(row["r"]))

    if hide_isolated:
        isolated = [node for node in graph.nodes() if graph.degree(node) == 0]
        graph.remove_nodes_from(isolated)

    if graph.number_of_nodes() == 0:
        return None, None

    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        k = 1.5 / max(1.0, np.sqrt(graph.number_of_nodes()))
        pos = nx.spring_layout(graph, seed=42, k=k)

    pos_edge_x, pos_edge_y = [], []
    neg_edge_x, neg_edge_y = [], []
    mid_x, mid_y, mid_text = [], [], []

    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        r = data["r"]
        if r >= 0:
            pos_edge_x += [x0, x1, None]
            pos_edge_y += [y0, y1, None]
        else:
            neg_edge_x += [x0, x1, None]
            neg_edge_y += [y0, y1, None]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_text.append(f"{u} × {v}<br>r = {r:.3f}")

    import plotly.graph_objects as go

    edge_trace_pos = go.Scatter(
        x=pos_edge_x,
        y=pos_edge_y,
        mode="lines",
        line=dict(width=1.5, color="rgba(200,30,30,0.55)"),
        hoverinfo="skip",
        name="正相關 (r > 0)",
    )
    edge_trace_neg = go.Scatter(
        x=neg_edge_x,
        y=neg_edge_y,
        mode="lines",
        line=dict(width=1.5, color="rgba(30,60,200,0.55)"),
        hoverinfo="skip",
        name="負相關 (r < 0)",
    )
    edge_hover_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        text=mid_text,
        showlegend=False,
    )

    degrees = dict(graph.degree())

    if node_size_basis == "總豐度 (Total abundance)" and abundance is not None:
        sizes_raw = {node: float(abundance.get(node, 0.0)) for node in graph.nodes()}
    else:
        sizes_raw = {node: float(degrees[node]) for node in graph.nodes()}

    max_size_raw = max(sizes_raw.values()) if sizes_raw else 1.0
    max_size_raw = max_size_raw if max_size_raw > 0 else 1.0

    node_list = list(graph.nodes())
    node_x = [pos[node][0] for node in node_list]
    node_y = [pos[node][1] for node in node_list]
    node_size = [10 + 30 * (sizes_raw[node] / max_size_raw) for node in node_list]
    node_hover = [f"{node}<br>degree = {degrees[node]}" for node in node_list]
    node_text = [node if len(node) <= 18 else node[:16] + "…" for node in node_list]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(size=node_size, color="#2b6cb0", line=dict(width=1, color="white")),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace_pos, edge_trace_neg, edge_hover_trace, node_trace])
    fig.update_layout(
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
    )

    hub_table = pd.DataFrame(
        {
            "taxon": node_list,
            "degree": [degrees[node] for node in node_list],
        }
    ).sort_values("degree", ascending=False).reset_index(drop=True)

    return fig, hub_table


# =========================
# Group comparison helpers
# =========================

def build_diff_table(
    corr_a: pd.DataFrame,
    p_a: pd.DataFrame,
    corr_b: pd.DataFrame,
    p_b: pd.DataFrame,
    fdr_alpha: float,
    name_a: str,
    name_b: str,
) -> pd.DataFrame:
    """
    Taxa pairs whose significance status (significant vs not, at the same
    FDR threshold) differs between two groups, sorted by |delta_r|
    descending. A quick way to spot co-occurrence relationships that
    appear in one condition (e.g. disease) but not the other (e.g.
    healthy).

    STATISTICAL CAVEAT, important if this feeds a publication: "significant
    in group A but not group B" is NOT itself a statistical test that the
    two correlations differ, and treating it as one is a well-known
    fallacy (Gelman & Stern, 2006, "The difference between 'significant'
    and 'not significant' is not itself statistically significant" -- e.g.
    r=0.50 (p=0.04) in a group of 20 vs r=0.45 (p=0.06) in another group of
    20 would show up here as "differs," even though those two correlations
    are barely distinguishable). This table is a fast screening/ranking
    tool, not a per-pair hypothesis test. For a defensible claim that a
    specific taxa pair's co-occurrence differs between groups, use a direct
    test for the difference between two correlation coefficients (e.g. a
    Fisher r-to-z comparison of the two group's correlations, which is not
    currently computed here) rather than citing this table's rows as
    significant differences on their own.
    """
    taxa = np.array(corr_a.index.astype(str))
    n = len(taxa)

    if n < 2:
        return pd.DataFrame()

    iu = np.triu_indices(n, k=1)

    r_a = corr_a.to_numpy(dtype=float)[iu]
    r_b = corr_b.to_numpy(dtype=float)[iu]
    p_a_v = p_a.to_numpy(dtype=float)[iu]
    p_b_v = p_b.to_numpy(dtype=float)[iu]

    sig_a = p_a_v <= fdr_alpha
    sig_b = p_b_v <= fdr_alpha
    differ = sig_a != sig_b

    out = pd.DataFrame(
        {
            "taxon_1": taxa[iu[0][differ]],
            "taxon_2": taxa[iu[1][differ]],
            f"r_{name_a}": r_a[differ],
            f"adj_p_{name_a}": p_a_v[differ],
            f"significant_{name_a}": sig_a[differ],
            f"r_{name_b}": r_b[differ],
            f"adj_p_{name_b}": p_b_v[differ],
            f"significant_{name_b}": sig_b[differ],
        }
    )

    if out.empty:
        return out

    out["abs_delta_r"] = (out[f"r_{name_a}"] - out[f"r_{name_b}"]).abs()
    out = out.sort_values("abs_delta_r", ascending=False).reset_index(drop=True)
    out = out.rename(columns={"abs_delta_r": "|delta_r|"})

    return out


# =========================
# Plot functions
# =========================

# On-screen figures never need to be rasterized above this DPI -- the
# browser downscales anyway. The user's chosen `dpi` (up to 600) is still
# honored, but only at export time (see the PNG/PDF savefig calls below),
# so cranking DPI up for a nice print-quality download no longer makes
# every interactive redraw slow too.
_PREVIEW_DPI_CAP = 150

# Hard safety ceiling for PNG/PDF *export* rendering. With "Auto figure size"
# on, the figure side can reach up to 120 cm for large taxa counts, and DPI
# defaults to 600 -- at that combination a raster buffer would be roughly
# 28,000 x 28,000 px (~3 GB just for one RGBA buffer), which reliably OOM-
# kills the process on Streamlit Community Cloud's free-tier memory limit
# (this is what "big tables crash the app" turned out to be: not a bug
# triggered by any specific taxa, but the eager, uncapped-DPI PNG+PDF export
# that used to run automatically on every "Run analysis", regardless of
# whether the user ever clicked download). This cap silently reduces the
# *effective* export DPI (never the on-screen preview, which already has
# its own cap above) so the exported raster's longer side never exceeds
# this many pixels, and the UI shows the user the DPI actually used.
_MAX_EXPORT_PIXELS_PER_SIDE = 12000


def _capped_export_dpi(fig_w_cm: float, fig_h_cm: float, requested_dpi: int) -> int:
    """Return the largest DPI <= requested_dpi that keeps both the width and
    height of the exported raster under _MAX_EXPORT_PIXELS_PER_SIDE pixels."""
    longer_side_in = max(fig_w_cm, fig_h_cm) / 2.54
    if longer_side_in <= 0:
        return requested_dpi
    max_dpi_for_cap = int(_MAX_EXPORT_PIXELS_PER_SIDE / longer_side_in)
    return max(72, min(requested_dpi, max_dpi_for_cap))


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
    """Draw heatmap (static, matplotlib/seaborn -- used for the main
    heatmap, per-group comparison heatmaps, and the PNG/PDF downloads)."""
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

    preview_dpi = min(dpi, _PREVIEW_DPI_CAP)

    # Scoped font-size override instead of the original matplotlib.rc(...),
    # which mutates *global* rcParams for the whole Python process. On a
    # shared Streamlit server that means one user's font-size setting could
    # leak into another user's session; rc_context() undoes it automatically
    # once this function returns.
    with plt.rc_context({"font.size": font_size}):
        fig, ax = plt.subplots(
            figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54),
            dpi=preview_dpi,
            constrained_layout=True,
        )

        sns.heatmap(
            plot_df,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            xticklabels=True,
            yticklabels=True,
            annot=False,
            linewidths=linewidths,
            linecolor="white",
            rasterized=True,  # keeps PDF export fast/small for many cells
            cbar_kws={
                "label": "Spearman correlation coefficient",
                "shrink": 0.75,
                "pad": 0.03,
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


_PLOTLY_CMAP_MAP = {
    "bwr_r": "RdBu",
    "coolwarm": "RdBu",
    "vlag": "RdBu",
    "icefire": "Tealrose",
    "RdBu_r": "RdBu_r",
    "viridis": "Viridis",
}


def draw_heatmap_plotly(
    corr_df_ord: pd.DataFrame,
    p_df_ord: pd.DataFrame,
    fdr_alpha: float,
    show_mode: str,
    cmap: str,
    shorten_plot_labels: bool,
    max_label_len: int,
    display_label_mode: str,
):
    """Interactive heatmap: same clustering order and significance mask
    as the static plot, but zoomable/pannable with exact r / adjusted-p
    values on hover. The static matplotlib plot remains the one used for
    PNG/PDF export."""
    import plotly.graph_objects as go

    if show_mode == "Show significant only":
        z_df = corr_df_ord.where(p_df_ord <= fdr_alpha)
    else:
        z_df = corr_df_ord.copy()

    labels = [format_display_label(i, mode=display_label_mode) for i in z_df.index]
    if shorten_plot_labels:
        labels = [shorten_label(i, max_len=max_label_len) for i in labels]

    z = z_df.to_numpy(dtype=float)
    p_vals = p_df_ord.to_numpy(dtype=float)
    colorscale = _PLOTLY_CMAP_MAP.get(cmap, "RdBu")

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            customdata=p_vals,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            zmid=0,
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<br>adj p = %{customdata:.3g}<extra></extra>",
            colorbar=dict(title="Spearman r"),
        )
    )
    fig.update_layout(
        xaxis=dict(tickangle=90),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
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

# All sidebar controls live inside a single st.form(). In the original app,
# every widget (including things unrelated to the maths, like "Grid line
# width") triggered a full script rerun that recomputed the correlation
# matrix, clustering and heatmap. Batching them behind one "Run analysis"
# button means you can change several settings and only pay the
# recomputation cost once, when you're ready.
with st.sidebar:
    st.header("📄 Upload")

    with st.form("controls_form"):
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

        st.caption(
            "⚠️ 上面的手動篩選是跟「Taxa label mode」轉換**之後**的名稱比對,"
            "不是原始 taxonomy 全路徑。例如 label mode 選 "
            "\"Use species-level only\" 時,要輸入 `Streptococcus_mitis`,"
            "而不是 `Bacteria|...|Streptococcus_mitis` 整串路徑,否則會完全比對不到、"
            "篩選後變成 0 筆。"
        )

        top_n_enabled = st.checkbox(
            "🔝 只保留總豐度前 N 名的 taxa",
            value=False,
            help=(
                "依照每個 taxon 在所有樣本中的豐度總和排序,只保留前 N 名。"
                "會在手動篩選之後套用,方便快速縮小到最主要的菌種,"
                "也能大幅減少畫熱圖/網路圖的負擔。"
            ),
        )

        top_n = st.number_input(
            "N",
            min_value=1,
            max_value=1000,
            value=20,
            step=1,
            disabled=not top_n_enabled,
        )

        st.header("🧫 資料前處理 (Normalization)")

        normalization_method = st.selectbox(
            "轉換方法",
            NORMALIZATION_METHODS,
            index=0,
            help=(
                "微生物體豐度資料具有「組成性」(compositional):同一樣本內所有 "
                "taxa 的總和固定,直接對 raw counts 做 Spearman 容易產生假的負相關"
                "(Aitchison, 1986; Gloor et al., 2017)。"
                "⚠️ TSS(相對豐度)只校正定序深度差異,並**沒有解決**組成性資料本身"
                "「總和固定」造成的假相關問題,而且因為每個樣本除以的分母(該樣本總和)"
                "不同,同一個 taxon 在不同樣本間的排序其實也會被改變,不是單純縮放。"
                "CLR (centered log-ratio) 才是文獻上處理組成性資料相關性分析的標準轉換,"
                "會用對數比值取代原始豐度,才真正打破「總和固定」造成的人為負相關結構。"
                "若要投稿發表,建議優先使用 CLR,並在方法段落說明所使用的轉換方式。"
                "預設為不轉換,與舊版行為一致。"
            ),
        )

        auto_pseudocount = st.checkbox(
            "CLR: 自動計算 pseudocount(建議)",
            value=True,
            help="使用資料中最小非零值的一半,避免 log(0)。只在選擇 CLR 時使用。",
        )

        manual_pseudocount = st.number_input(
            "CLR: 手動 pseudocount(若取消自動計算)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help="只在取消勾選「自動計算」且選擇 CLR 轉換時使用。",
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
                "ward",
            ],
            index=0,
            help=(
                "Average linkage is recommended for correlation-distance "
                "heatmaps. 'ward' is included because it's what the README's "
                "CLI-aligned description mentions; the default here stays "
                "'average' to match this app's previous behavior."
            ),
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
            value=30.0,
            step=1.0,
        )

        font_size = st.number_input(
            "Font size",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
        )

        dpi = st.number_input(
            "DPI",
            min_value=72,
            max_value=600,
            value=600,
            step=10,
            help=(
                "Applied to the PNG/PDF downloads. The on-screen preview is "
                f"capped at {_PREVIEW_DPI_CAP} DPI for speed regardless of "
                "this setting."
            ),
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

        st.header("🖥️ 互動式熱圖")

        show_interactive_heatmap = st.checkbox(
            "顯示互動式熱圖 (Plotly)",
            value=False,
            help="可以滑鼠縮放、平移,游標移到格子上會顯示精確的 r 與校正後 p 值。",
        )

        st.header("🕸️ 共現網路圖")

        show_network = st.checkbox(
            "顯示共現網路圖 (Co-occurrence network)",
            value=False,
            help="節點 = taxa,邊 = 通過顯著性門檻的相關性。",
        )

        network_layout = st.selectbox(
            "網路圖排列方式",
            ["spring", "kamada_kawai", "circular"],
            index=0,
        )

        hide_isolated_nodes = st.checkbox(
            "隱藏沒有顯著相關的孤立節點",
            value=True,
        )

        node_size_basis = st.selectbox(
            "節點大小依據",
            ["連結數 (Degree)", "總豐度 (Total abundance)"],
            index=0,
        )

        st.header("👥 分組比較（選用）")

        metadata_file = st.file_uploader(
            "上傳分組 metadata 表 (.tsv/.csv, 選用)",
            type=["tsv", "txt", "csv"],
            help=(
                "第一欄需為樣本 ID,需與豐度表的樣本欄位名稱一致。"
                "其餘欄位可以是分組資訊(例如 health/disease)。"
            ),
            key="metadata_uploader",
        )

        metadata_df = None
        group_col = None

        if metadata_file is not None:
            try:
                meta_sep = _infer_sep(getattr(metadata_file, "name", ""))
                metadata_df = pd.read_csv(metadata_file, sep=meta_sep)
                metadata_df.columns = [str(c).strip() for c in metadata_df.columns]
                sample_id_col = metadata_df.columns[0]
                metadata_df[sample_id_col] = metadata_df[sample_id_col].astype(str).str.strip()

                candidate_cols = list(metadata_df.columns[1:])
                if candidate_cols:
                    group_col = st.selectbox(
                        "選擇分組欄位",
                        candidate_cols,
                        index=0,
                        key="group_col_select",
                    )
                else:
                    st.warning("Metadata 檔案只有一欄,無法選擇分組欄位。")
            except Exception as exc:
                st.warning(f"無法讀取 metadata 檔案:{exc}")
                metadata_df = None

        st.form_submit_button("🚀 Run analysis", width="stretch")


# =========================
# Main analysis
# =========================

if uploaded is None:
    st.info("Upload a table from the sidebar, then click **Run analysis**.")
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
        width="stretch",
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
            width="stretch",
        )

if df.empty:
    st.warning(
        "No taxa/features available after filtering. "
        "Please check bacteria names, label mode, or table format."
    )
    st.stop()

if top_n_enabled:
    total_abundance = df.sum(axis=1)
    keep_index = total_abundance.sort_values(ascending=False).head(int(top_n)).index
    df = df.loc[keep_index]
    st.info(
        f"🔝 只保留總豐度前 {int(top_n)} 名 taxa "
        f"(依所有樣本加總後的豐度排序);目前剩下 {df.shape[0]} 筆。"
    )

if df.shape[0] < 2:
    st.warning(
        "At least two taxa/features are required for correlation analysis."
    )
    st.stop()

if df.shape[1] < 3:
    st.warning(
        "At least three samples are required to compute a Spearman "
        "correlation at all."
    )
    st.stop()

if df.shape[1] < 10:
    st.warning(
        f"Only {df.shape[1]} samples. Spearman's p-value here uses scipy's "
        "asymptotic (t-distribution) approximation regardless of sample "
        "size, which is unreliable for very small n -- with n=3, no "
        "correlation can reach p < 0.05 two-tailed even at r = ±1 (the "
        "smallest possible two-tailed p-value is 1/3). Results from this "
        "few samples are exploratory; if this is going into a publication, "
        "note the small n as a limitation, or collect more samples before "
        "drawing conclusions from significance."
    )


# Auto figure size
n_taxa = df.shape[0]

if n_taxa > 300:
    st.warning(
        f"{n_taxa} taxa/features selected. A heatmap this large will be slow "
        "to render/export and hard to read. Consider narrowing with the "
        "taxa filter, or switching to genus-level grouping."
    )

if auto_fig_size:
    fig_side = max(20.0, min(120.0, n_taxa * 0.75 + 10))
    fig_w = fig_side
    fig_h = fig_side
else:
    fig_w = manual_fig_w
    fig_h = manual_fig_h


# Preview tables
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🧾 Taxa/features used for analysis")
    st.dataframe(
        pd.DataFrame({"Taxa / Feature": df.index}),
        width="stretch",
        height=360,
    )

with right_col:
    st.subheader("📋 Processed abundance table preview")
    st.dataframe(
        df.iloc[:30, :10],
        width="stretch",
        height=360,
    )


# Preprocessing (normalization) applied before correlation
if normalization_method == NORM_CLR:
    pseudocount = suggest_pseudocount(df) if auto_pseudocount else float(manual_pseudocount)
    st.caption(f"CLR pseudocount used: {pseudocount:.6g}")
else:
    pseudocount = 0.0

df_analysis = normalize_table(df, normalization_method, pseudocount)

if normalization_method != NORM_RAW:
    with st.expander("🧫 轉換後的資料預覽 (用於相關性分析)", expanded=False):
        st.dataframe(df_analysis.iloc[:30, :10], width="stretch")


# Correlation analysis
with st.spinner("Calculating Spearman correlations..."):
    corr_df, p_df = spearman_corr_and_p(
        df_analysis,
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


def _build_static_fig() -> plt.Figure:
    """Build the static matplotlib heatmap. Factored out so it can be
    skipped entirely (not just hidden) when the interactive Plotly heatmap
    is showing the same clustered matrix, and built lazily later only if
    the user actually asks for a PNG/PDF export -- see the export button
    below. Rendering this is one of the more expensive steps for a large
    heatmap (seconds, even at the capped preview DPI), so previously
    computing and displaying it unconditionally meant every "Run analysis"
    paid that cost even when the interactive view made it redundant."""
    return draw_heatmap(
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


fig = None  # built below, or lazily by the export button further down

if show_interactive_heatmap:
    st.caption(
        "互動式熱圖(下方)已取代這裡的靜態預覽,避免同一張圖畫兩次;"
        "如需下載 PNG/PDF,仍可在下方按鈕產生(用同一組聚類結果)。"
    )
else:
    fig = _build_static_fig()
    st.pyplot(
        fig,
        clear_figure=False,
        width="stretch",
    )

st.caption(
    f"Figure size: {fig_w:.1f} cm × {fig_h:.1f} cm | "
    f"Taxa/features: {df.shape[0]} | "
    f"Samples: {df.shape[1]} | "
    f"Display mode: {show_mode} | "
    f"Label display: {display_label_mode} | "
    f"Normalization: {normalization_method}"
)

if show_interactive_heatmap:
    st.subheader("🖥️ 互動式熱圖 (Plotly)")
    interactive_fig = draw_heatmap_plotly(
        corr_df_ord=corr_df_ord,
        p_df_ord=p_df_ord,
        fdr_alpha=fdr_alpha,
        show_mode=show_mode,
        cmap=cmap,
        shorten_plot_labels=shorten_plot_labels,
        max_label_len=max_label_len,
        display_label_mode=display_label_mode,
    )
    st.plotly_chart(interactive_fig, width="stretch")
    st.caption("游標移到格子上可看到精確的 r 與校正後 p 值;可滑鼠滾輪縮放、拖曳平移。")


# Result matrices
with st.expander("📊 Correlation matrix, clustered order", expanded=False):
    st.dataframe(
        corr_df_ord,
        width="stretch",
    )

with st.expander("📊 Adjusted P-value matrix, same order", expanded=False):
    st.dataframe(
        p_df_ord,
        width="stretch",
    )


# =========================
# Downloads
# =========================

st.subheader("⬇️ Download results")

# A plain-text summary of every setting that affects the numbers above --
# for reproducibility / writing an accurate methods section. None of this
# is re-derived from the data, it's just a record of what was actually
# selected in the sidebar for this run.
_params_lines = [
    "MicroCoHeat analysis parameters",
    "================================",
    f"Taxa/features analyzed: {df.shape[0]}",
    f"Samples analyzed: {df.shape[1]}",
    f"Taxa label mode: {label_mode}",
    f"Manual taxa filter: {'(none)' if not taxa_list else f'{len(taxa_list)} term(s), {match_mode}, case_sensitive={case_sensitive}'}",
    f"Top-N by abundance filter: {f'top {int(top_n)}' if top_n_enabled else '(not applied)'}",
    f"Normalization: {normalization_method}"
    + (f" (pseudocount={pseudocount:.6g})" if normalization_method == NORM_CLR else ""),
    f"Correlation method: Spearman",
    f"P-value adjustment: {p_adjust_method}",
    f"FDR alpha: {fdr_alpha}",
    f"Clustering linkage: {cluster_method}",
    f"Heatmap display mode: {show_mode}",
]
st.download_button(
    "Download analysis parameters (.txt)",
    data=("\n".join(_params_lines) + "\n").encode("utf-8"),
    file_name="microcoheat_analysis_parameters.txt",
    mime="text/plain",
    help=(
        "A record of every setting used for this run (normalization, FDR "
        "method/alpha, clustering, filters) -- for reproducing the analysis "
        "or writing the methods section."
    ),
)

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

if normalization_method != NORM_RAW:
    csv_normalized = df_analysis.to_csv().encode("utf-8-sig")
    st.download_button(
        f"Download normalized table CSV ({normalization_method})",
        data=csv_normalized,
        file_name="microcoheat_normalized_table.csv",
        mime="text/csv",
    )

st.markdown("**🖼️ Heatmap PNG / PDF export**")

effective_export_dpi = _capped_export_dpi(fig_w, fig_h, dpi)
if effective_export_dpi < dpi:
    st.caption(
        f"⚠️ Requested {dpi} DPI at this figure size would need a raster "
        f"over {_MAX_EXPORT_PIXELS_PER_SIDE:,}px on a side (too much memory "
        f"for the free-tier server). Export DPI automatically reduced to "
        f"{effective_export_dpi} for the PNG/PDF download below; the "
        f"on-screen heatmap above is unaffected."
    )

# Export signature: only these inputs affect the rendered PNG/PDF, so the
# cached bytes below stay valid across reruns caused by clicking a download
# button itself, and are only recomputed when something that would actually
# change the image changes.
_export_sig = (
    tuple(corr_df_ord.columns),
    show_mode,
    cmap,
    fig_w,
    fig_h,
    font_size,
    effective_export_dpi,
    linewidths,
    shorten_plot_labels,
    max_label_len,
    display_label_mode,
)

# Generating both a PNG and a PDF at export quality is the single most
# expensive step in the whole app for a large heatmap (previously ran
# unconditionally on every "Run analysis", which is what made big tables
# crash the deployed app -- see _MAX_EXPORT_PIXELS_PER_SIDE above). Gating
# it behind an explicit button means a user who only wants to look at the
# on-screen heatmap, or download the CSVs, never pays that cost.
if st.button("🖼️ Generate PNG / PDF for download", key="generate_heatmap_export"):
    with st.spinner("Rendering export-quality PNG/PDF..."):
        # fig is only pre-built above when the interactive heatmap is off;
        # when it's on, this is the first (and only) time the static
        # matplotlib figure gets rendered at all.
        if fig is None:
            fig = _build_static_fig()

        buf_png = io.BytesIO()
        buf_pdf = io.BytesIO()

        fig.savefig(
            buf_png,
            format="png",
            dpi=effective_export_dpi,
            bbox_inches="tight",
        )
        fig.savefig(
            buf_pdf,
            format="pdf",
            dpi=effective_export_dpi,
            bbox_inches="tight",
        )

        st.session_state["heatmap_export_sig"] = _export_sig
        st.session_state["heatmap_export_png"] = buf_png.getvalue()
        st.session_state["heatmap_export_pdf"] = buf_pdf.getvalue()

if st.session_state.get("heatmap_export_sig") == _export_sig:
    st.download_button(
        "Download heatmap PNG",
        data=st.session_state["heatmap_export_png"],
        file_name="microcoheat_heatmap.png",
        mime="image/png",
    )

    st.download_button(
        "Download heatmap PDF",
        data=st.session_state["heatmap_export_pdf"],
        file_name="microcoheat_heatmap.pdf",
        mime="application/pdf",
    )
elif "heatmap_export_sig" in st.session_state:
    st.caption("Settings changed since the last export -- click the button above to regenerate.")

if fig is not None:
    plt.close(fig)


# =========================
# Co-occurrence network
# =========================

if show_network:
    st.subheader("🕸️ 共現網路圖 (Co-occurrence network)")

    edge_df = build_edge_table(corr_df_ord, p_df_ord, fdr_alpha)

    if edge_df.empty:
        st.info("目前的顯著性門檻下沒有偵測到任何顯著相關的 taxa pair,無法繪製網路圖。")
    else:
        abundance_map = df.sum(axis=1).to_dict()
        net_fig, hub_table = draw_network_figure(
            edge_df,
            all_taxa=[str(t) for t in df_analysis.index],
            abundance=abundance_map,
            layout=network_layout,
            hide_isolated=hide_isolated_nodes,
            node_size_basis=node_size_basis,
        )

        if net_fig is not None:
            st.plotly_chart(net_fig, width="stretch")
            st.caption(
                f"共 {len(edge_df)} 條顯著共現關係 (adj p ≤ {fdr_alpha})。"
                f"節點大小依「{node_size_basis}」縮放。"
            )
        else:
            st.info("勾選了隱藏孤立節點,且目前沒有任何節點有顯著相關,因此沒有東西可畫。")

        with st.expander("🔗 顯著共現關係列表 (edge table)", expanded=False):
            st.dataframe(edge_df, width="stretch")
            st.download_button(
                "下載共現關係 CSV (Cytoscape / Gephi 可用)",
                data=edge_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="microcoheat_network_edges.csv",
                mime="text/csv",
            )

        if hub_table is not None and not hub_table.empty:
            with st.expander("⭐ Hub taxa (依連結數排序)", expanded=False):
                st.dataframe(hub_table, width="stretch")
                st.download_button(
                    "下載 Hub taxa CSV",
                    data=hub_table.to_csv(index=False).encode("utf-8-sig"),
                    file_name="microcoheat_hub_taxa.csv",
                    mime="text/csv",
                )


# =========================
# Group comparison
# =========================

if metadata_df is not None and group_col is not None:
    sample_id_col = metadata_df.columns[0]
    sample_to_group = dict(
        zip(metadata_df[sample_id_col], metadata_df[group_col].astype(str).str.strip())
    )

    sample_cols = [str(c) for c in df_analysis.columns]
    matched = [s for s in sample_cols if s in sample_to_group]
    unmatched = [s for s in sample_cols if s not in sample_to_group]

    st.subheader("👥 組間比較 (Group comparison)")

    if unmatched:
        preview = ", ".join(unmatched[:10]) + ("..." if len(unmatched) > 10 else "")
        st.warning(f"有 {len(unmatched)} 個樣本在 metadata 中找不到對應分組,已略過: {preview}")

    groups: Dict[str, List[str]] = {}
    for sample in matched:
        groups.setdefault(sample_to_group[sample], []).append(sample)

    valid_groups = {g: cols for g, cols in groups.items() if len(cols) >= 3}
    skipped_groups = [g for g, cols in groups.items() if len(cols) < 3]

    if skipped_groups:
        st.info(f"以下分組樣本數 < 3,已略過分析: {', '.join(skipped_groups)}")

    if len(valid_groups) < 2:
        st.info("目前樣本數足夠 (≥3) 的分組少於兩組,至少需要兩組才能比較。")
    else:
        group_names = list(valid_groups.keys())
        group_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        overall_order = corr_df_ord.index.tolist()

        tabs = st.tabs(group_names)
        for tab, gname in zip(tabs, group_names):
            with tab:
                sub_df = df_analysis[valid_groups[gname]]
                g_corr, g_p = spearman_corr_and_p(
                    sub_df, fdr_alpha=fdr_alpha, method=p_adjust_method
                )
                g_corr = g_corr.reindex(index=overall_order, columns=overall_order)
                g_p = g_p.reindex(index=overall_order, columns=overall_order)
                group_results[gname] = (g_corr, g_p)

                g_fig = draw_heatmap(
                    corr_df_ord=g_corr,
                    p_df_ord=g_p,
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
                st.pyplot(g_fig, clear_figure=False, width="stretch")
                st.caption(f"{gname}: {len(valid_groups[gname])} 個樣本")
                plt.close(g_fig)

        col_a, col_b = st.columns(2)
        with col_a:
            group_a = st.selectbox("比較組 A", group_names, index=0, key="diff_group_a")
        with col_b:
            default_b = 1 if len(group_names) > 1 else 0
            group_b = st.selectbox("比較組 B", group_names, index=default_b, key="diff_group_b")

        if group_a == group_b:
            st.caption("請選擇兩個不同的分組來比較。")
        else:
            corr_a, p_a = group_results[group_a]
            corr_b, p_b = group_results[group_b]
            diff_table = build_diff_table(corr_a, p_a, corr_b, p_b, fdr_alpha, group_a, group_b)

            if diff_table.empty:
                st.caption(f"{group_a} 與 {group_b} 之間沒有偵測到顯著性不同的 taxa pair。")
            else:
                st.write(f"**{group_a} vs {group_b}**:顯著性不同的 taxa pair(依 |Δr| 排序):")
                st.caption(
                    "⚠️ 這張表是「A 組顯著、B 組不顯著」(或反之)的篩選結果,"
                    "**不是**兩組相關係數差異的統計檢定——這是統計上有名的謬誤"
                    "(顯著 vs 不顯著,兩者本身的差距不一定顯著;Gelman & Stern, 2006)。"
                    "適合用來快速篩選/排序候選 taxa pair,若要在論文中主張某個 pair "
                    "在兩組間「顯著不同」,需要另外對兩個相關係數做直接比較的檢定"
                    "(例如 Fisher r-to-z),而不是引用這張表本身。"
                )
                st.dataframe(diff_table.head(200), width="stretch")
                st.download_button(
                    f"下載差異表 CSV ({group_a}_vs_{group_b})",
                    data=diff_table.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"microcoheat_diff_{group_a}_vs_{group_b}.csv",
                    mime="text/csv",
                )
