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

> **Note on "Oh no. Error running app.":** Streamlit Community Cloud has a
> known, currently-unresolved bug where it ignores `runtime.txt` and always
> builds with a newer Python than requested. If that newer Python has no
> installable wheel for an old, tightly-pinned dependency (e.g. `numpy<2`),
> the deploy fails with this generic error page and no useful traceback.
> `requirements.txt` in this repo deliberately does **not** cap `numpy`/`pandas`
> at `<2`/`<3` for this reason — the app is tested against both dependency
> lines. If you ever see this error again, check **Manage app → Logs** on
> Streamlit Cloud first; a `ResolutionImpossible` or build failure during
> "Processing dependencies" points at a version pin, not application code.

## 📄 Input
- `genus-table.tsv` or `.csv` with the **first column as taxonomy/feature ID** (index).
- Filtering preserves duplicate genera (strict last-rank match), matching the CLI workflow.

## 🔎 Taxa filtering
- **Taxa label mode** transforms the index (e.g. "Use species-level only" keeps just `Streptococcus_mitis` from a full `Bacteria|...|Streptococcus_mitis` path) *before* anything else runs, including the manual filter below.
- **Manual filter** ("Enter bacteria names") matches against the label **after** that transform, not the original taxonomy string — a common source of "0 taxa after filtering": pasting the full path when the label mode has already shortened it to just the species/genus name. Match against whatever the label mode produces, or switch to "Use table labels as-is" to match full paths.
- **🔝 Only keep the top N most abundant taxa** (new) — an alternative to typing names by hand: ranks taxa by total abundance summed across all samples and keeps only the top N, applied after the manual filter. Useful both to focus on the dominant taxa and to cut a very large table down to a size that's fast to cluster/render (see the performance notes below on why large tables used to crash the deployed app).

## 🧪 Method
- Spearman with `axis=1` (vectorized: the whole correlation/p-value matrix is computed in one call, not pair-by-pair)
- Benjamini–Hochberg FDR (`fdr_bh` by default; `bonferroni`, `holm`, `fdr_by`, `sidak`, `holm-sidak` and `ward`-eligible clustering are also selectable)
- Mask non-significant entries in the *displayed* heatmap (does not affect the underlying clustering, which always uses the full correlation matrix — see "Known discrepancies" below)
- Hierarchical clustering on `1 - r` after symmetrization and setting diagonal = 1; default linkage is `average`, `ward` is also available in the sidebar

## 🧫 Optional preprocessing (new)
Microbiome abundance data is *compositional* (every sample's taxa necessarily sum to a fixed total), which can make Spearman correlation on raw counts produce spurious relationships (Aitchison, 1986; Gloor et al., 2017, "Microbiome Datasets Are Compositional"). The sidebar now offers:
- **Raw values** — no transform (default; matches all previous behavior).
- **Relative abundance (TSS)** — divide each sample by its own total. This only corrects for sequencing-depth differences; it does **not** resolve the compositional "constant sum" artifact, and because each sample is divided by a different total, TSS can still change a given taxon's rank order across samples (it isn't a pure rescaling relative to raw counts, any more than CLR is).
- **CLR (centered log-ratio)** — the transform actually recommended in the compositional-data literature for this, since it replaces raw abundances with log-ratios to each sample's own geometric mean, which is what breaks the constant-sum artifact. A pseudocount (auto-suggested as half the smallest non-zero value in the table, or set manually) avoids `log(0)`. **If this analysis is going into a publication, CLR is the defensible default to report** — state whichever method was actually used in the methods section.

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

## ⚠️ Statistical caveats (read before using this for a publication)
- **The group-comparison diff table is a screening tool, not a hypothesis test.** "Significant in group A, not significant in group B" is not itself a test that the two correlations differ — two correlations that are barely distinguishable (e.g. r=0.50, p=0.04 vs r=0.45, p=0.06) will show up as "differing" here (Gelman & Stern, 2006). Use it to rank/shortlist candidate pairs, not to cite a row as a significant between-group difference. A defensible claim needs a direct test for the difference between two correlation coefficients (e.g. Fisher r-to-z), which this app does not currently compute.
- **Undefined correlations inside a group are silently set to r=0, adj_p=1.** A taxon that is invariant within one group's subset (even if it varies overall) makes Spearman's r/p undefined (NaN) for every pair involving it in that group; these are treated as "not significant" rather than excluded. Worth checking for and disclosing if a group has taxa with zero within-group variance.
- **Very small sample sizes are mathematically limited, not just underpowered.** With n=3 samples, no correlation can reach p < 0.05 two-tailed even at r = ±1 (the smallest possible two-tailed p-value is 1/3), and scipy's p-value uses an asymptotic (t-distribution) approximation regardless of n, which is unreliable for small samples generally. The app warns below 10 samples; treat results from very small groups as exploratory.
- **Reproducibility:** "⬇️ Download results" includes a `microcoheat_analysis_parameters.txt` with every setting used for that run (normalization + pseudocount, FDR method/alpha, clustering linkage, taxa filters, label mode) — useful for an accurate methods section or for reproducing a figure later.

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
- **PNG/PDF export is on-demand**, behind a "🖼️ Generate PNG / PDF for download" button, instead of running automatically on every "Run analysis". With "Auto figure size" on and a large taxa count, the figure side can reach 120 cm; at the default 600 DPI that would need a ~28,000 px raster (~3 GB just for one image buffer), which reliably crashed the app on Streamlit Community Cloud's free-tier memory limit for anyone with a few hundred+ taxa — this was the actual cause of "large tables always break it," not a data-specific bug. The export DPI is also automatically capped (`_MAX_EXPORT_PIXELS_PER_SIDE`, currently 12,000 px/side) regardless of your chosen DPI, with the actual DPI used shown in the UI if it had to be reduced.
