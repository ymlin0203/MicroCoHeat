# MicroCoHeat — Microbial Co-occurrence Heatmap

**MicroCoHeat** is a Streamlit-based platform for **microbial co-occurrence** analysis: a Spearman correlation heatmap, an optional interactive/network view, and an optional group (e.g. health vs. disease) comparison.

## 🚀 Local run
```bash
pip install -r requirements.txt
streamlit run app_microcoheat.py
```

## ☁️ Streamlit Community Cloud
1. Push this folder to GitHub.
2. In Streamlit Cloud: New app → Select repo/branch → **Main file path**: `app_microcoheat.py`.
3. Deploy. No secrets required.

> **Note on cold starts:** Streamlit Community Cloud reinstalls every package in
> `requirements.txt` from scratch whenever an idle app wakes up. This app's
> dependency list is heavier than a minimal Streamlit app (matplotlib,
> seaborn, scipy, statsmodels, networkx, plotly), so a cold wake-up commonly
> takes on the order of 1–3 minutes. That's install time, not this app's own
> startup code — `streamlit run` on an already-warm environment (local run,
> or an app that hasn't gone to sleep) reaches the upload screen in well
> under a second, because the heavier optional dependencies (`statsmodels`,
> `plotly`, `networkx`) are only imported the first time you actually use
> the feature that needs them, not at startup.

## 📄 Input
- `genus-table.tsv` or `.csv` with the **first column as taxonomy/feature ID** (index).
- Filtering preserves duplicate genera (strict last-rank match), matching the CLI workflow.

## 🧪 Method
- Spearman with `axis=1` (vectorized: the whole correlation/p-value matrix is computed in one call, not pair-by-pair)
- Benjamini–Hochberg FDR (`fdr_bh` by default; `bonferroni`, `holm`, `fdr_by`, `sidak`, `holm-sidak` and `ward`-eligible clustering are also selectable)
- Mask non-significant entries in the *displayed* heatmap (does not affect the underlying clustering, which always uses the full correlation matrix — see "Known discrepancies" below)
- Hierarchical clustering on `1 - r` after symmetrization and setting diagonal = 1; default linkage is `average`, `ward` is also available in the sidebar

## 🧫 Optional preprocessing (new)
Microbiome abundance data is *compositional* (every sample's taxa necessarily sum to a fixed total), which can make Spearman correlation on raw counts produce spurious relationships. The sidebar now offers:
- **Raw values** — no transform (default; matches all previous behavior).
- **Relative abundance (TSS)** — divide each sample by its own total.
- **CLR (centered log-ratio)** — the standard compositional-data transform. Unlike TSS, this changes the *rank order across samples* for a given taxon (the per-sample geometric mean differs sample to sample), so it can genuinely change which correlations come out significant, not just rescale them. A pseudocount (auto-suggested as half the smallest non-zero value in the table, or set manually) avoids `log(0)`.

## 🖥️ Optional interactive heatmap (new)
Alongside the static (downloadable) heatmap, you can enable a Plotly version with the same clustering order and significance mask — zoomable/pannable, with exact `r` and adjusted `p` on hover.

## 🕸️ Optional co-occurrence network (new)
Turns the same significant pairs used in the heatmap into a network graph (nodes = taxa, edges = significant correlations, colored by sign). Includes:
- Choice of layout (`spring`, `kamada_kawai`, `circular`), and an option to hide taxa with no significant partner.
- Node size by degree or by total abundance.
- Downloadable edge table (`taxon_1, taxon_2, r, adj_p`) ready for Cytoscape/Gephi.
- Downloadable hub-taxa table (sorted by degree), to spot candidate keystone taxa.

## 👥 Optional group comparison (new)
Upload a metadata table (first column = sample ID, matching the abundance table's sample columns; any other column can be chosen as the grouping variable). For every group with ≥3 samples, MicroCoHeat computes its own correlation matrix (aligned to the same taxon order as the overall heatmap, for visual comparability) and shows it in its own tab. Pick any two groups to get a table of taxa pairs whose significance status differs between them (e.g. present in Disease, absent in Healthy), sorted by `|Δr|`, with a CSV download.

## 🖼️ Output
- Heatmap (PNG/PDF, static) and, optionally, an interactive Plotly heatmap
- Cluster-ordered correlation matrix CSV and adjusted P-value matrix CSV
- Optional: normalized table CSV, network edge table CSV, hub-taxa CSV, group-comparison CSV

## 🧷 Demo data
- `data/sample_genus_table.tsv` — a small synthetic 15-genus × 12-sample table with a couple of deliberate co-occurrence signals baked in, so the heatmap/network aren't empty on first try.
- `data/sample_metadata.tsv` — matching Healthy/Disease grouping for the same 12 samples, to try the group-comparison feature.

## ⚠️ Known discrepancies (documented, not silently changed)
Two things this README previously implied but the app did not (and still does not, by default) do, kept as-is rather than changed unilaterally since they affect the scientific result:
- Clustering always runs on the **full** correlation matrix, not the significance-masked one (masking is a display-only step, applied after clustering). If you specifically need "cluster on the masked matrix" for reproducing a particular pipeline, that would need a separate toggle.
- `ward` linkage is now a selectable option in the sidebar (it wasn't previously, despite being mentioned above), but the default remains `average` to avoid silently changing existing results for anyone already using this app. Pick `ward` explicitly if you want it.

## 🛠️ Performance notes (for anyone reading the code)
- Spearman correlation is computed with one vectorized `scipy.stats.spearmanr(values, axis=1)` call instead of a Python loop over every taxon pair (100–200x faster at 50–200 taxa).
- `statsmodels`, `plotly` and `networkx` are imported lazily, inside the functions that use them, so the app doesn't pay their import cost unless you actually use FDR correction / the interactive heatmap / the network graph.
- The on-screen heatmap is rendered at a capped preview DPI; your chosen DPI (up to 600) is only applied when generating the PNG/PDF downloads.
- Hierarchical-clustering distance matrices are built as plain NumPy arrays rather than mutating a DataFrame's `.values` in place, which avoids a `ValueError: underlying array is read-only` under pandas' Copy-on-Write mode (the default since pandas 3.0).
