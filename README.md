# MicroCoHeat — Microbial Co-occurrence Heatmap

**MicroCoHeat** is a Streamlit-based platform for **microbial co-occurrence heatmap** analysis.  
It implements a CLI-aligned framework: Spearman (rows-as-variables), BH FDR (alpha=0.05), mask non-significant correlations, and hierarchical clustering (Ward on 1 - r).

## 🚀 Local run
```bash
pip install -r requirements.txt
streamlit run app_microcoheat.py
```

## ☁️ Streamlit Community Cloud
1. Push this folder to GitHub (e.g., `ym-lin/MicroCoHeat`).
2. In Streamlit Cloud: New app → Select repo/branch → **Main file path**: `app_microcoheat.py`.
3. Deploy. No secrets required.

## 📄 Input
- `genus-table.tsv` or `.csv` with the **first column as taxonomy/feature ID** (index).
- Filtering preserves duplicate genera (strict last-rank match), matching the CLI workflow.

## 🧪 Method
- Spearman with `axis=1`
- Benjamini–Hochberg FDR (`fdr_bh`), α=0.05
- Mask non-significant entries (set to 0) **before** clustering
- Ward clustering on `1 - r` after symmetrization and setting diagonal=1

## 🖼️ Output
- Heatmap (PNG/PDF)
- Cluster-ordered correlation matrix CSV
- Adjusted P-value matrix CSV

## 🧷 Demo data
A tiny demo table is available at `data/sample_genus_table.tsv`.
