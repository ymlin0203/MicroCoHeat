# app_microcoheat.py
# Usage: streamlit run app_microcoheat.py

import io
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ========= Defaults (CLI-aligned) =========
DEFAULT_GENERA = [
    # OSCC
    'Peptostreptococcus','Catonella','Fusobacterium','Oribacterium','Treponema',
    'Aggregatibacter','Selenomonas','Capnocytophaga','Alloprevotella','Filifactor',
    'Parvimonas','Dialister','Atopobium','Prevotella','Solobacterium',
    # Normal / Precancer
    'Streptococcus','Halomonas','Corynebacterium','Nitratireductor','P5D1_392',
    'Pseudomonas','Saccharimonadaceae','Gordonia',
]

def _last_token(x: str) -> str:
    return str(x).split('|')[-1].strip()

def _infer_sep(name: str) -> str:
    ext = Path(name).suffix.lower()
    return ',' if ext == '.csv' else '\t'

@st.cache_data(show_spinner=False)
def read_table(src) -> pd.DataFrame:
    if isinstance(src, (str, Path)):
        sep = _infer_sep(str(src))
        return pd.read_csv(src, sep=sep, index_col=0)
    else:
        name = getattr(src, 'name', '') or ''
        sep = _infer_sep(name)
        return pd.read_csv(src, sep=sep, index_col=0)

def filter_by_genus_cli_exact(df: pd.DataFrame, genera: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    df2['Genus'] = [_last_token(i) for i in df2.index]
    df2 = df2[df2['Genus'].isin(genera)].drop(columns='Genus')
    df2.index = pd.Index([_last_token(i) for i in df2.index])
    return df2

@st.cache_data(show_spinner=False)
def spearman_corr_and_p(df: pd.DataFrame, axis: int, fdr_alpha: float, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr_mat, p_mat = spearmanr(df, axis=axis)
    n = len(df.index)
    pvals = p_mat.reshape(n**2)
    _rej, pvals_corr, _, _ = multipletests(pvals, alpha=fdr_alpha, method=method, is_sorted=False, returnsorted=False)
    p_corr = pvals_corr.reshape((n, n))
    taxa = df.index
    return (pd.DataFrame(corr_mat, index=taxa, columns=taxa),
            pd.DataFrame(p_corr, index=taxa, columns=taxa))

@st.cache_data(show_spinner=False)
def reorder_by_clustering_cli(corr_df_masked: pd.DataFrame, p_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr_df = (corr_df_masked + corr_df_masked.T) / 2
    for i in range(corr_df.shape[0]):
        corr_df.iat[i, i] = 1.0
    dist_array = squareform(1 - corr_df)
    Z = linkage(dist_array, 'ward')
    order = leaves_list(Z)
    return corr_df.iloc[order, order], p_df.iloc[order, order]

# ========= UI =========
st.set_page_config(page_title='MicroCoHeat — Microbial Co-occurrence Heatmap', layout='wide')
st.title('🧬 MicroCoHeat — Microbial Co-occurrence Heatmap (Spearman)')

with st.sidebar:
    st.header('⚙️ Settings')
    uploaded = st.file_uploader('📄 Upload genus-table (.tsv/.csv; first column = Taxon/Feature)', type=['tsv','txt','csv'])
    path_text = st.text_input('Or paste a local path (.tsv/.csv; choose one)', value='')

    cli_mode = st.checkbox('🧰 CLI-aligned mode (recommended)', value=True)

    st.subheader('🔎 Genus filter')
    genera_input = st.text_area('Custom list (comma-separated)', value=','.join(DEFAULT_GENERA), height=120)
    genera_list = [g.strip() for g in genera_input.split(',') if g.strip()]

    st.subheader('📐 Statistics')
    fdr_alpha       = 0.05 if cli_mode else st.number_input('FDR α', 0.0, 1.0, 0.05, 0.01)
    p_adjust_method = 'fdr_bh' if cli_mode else st.selectbox('P-value correction', ['fdr_bh','bonferroni','holm','fdr_by','sidak','holm-sidak'])
    axis_choice     = 1

    st.subheader('🎨 Plot')
    cmap       = 'bwr_r' if cli_mode else st.selectbox('Colormap', ['bwr_r','coolwarm','vlag','icefire','viridis','RdBu_r'])
    linewidths = 0.5     if cli_mode else st.number_input('Line width', 0.0, 5.0, 0.5, 0.1)
    font_size  = 7       if cli_mode else st.number_input('Font size', 4, 18, 7)
    fig_w      = 13.0    if cli_mode else st.number_input('Width (cm)', 6.0, 40.0, 13.0, 0.5)
    fig_h      = 11.0    if cli_mode else st.number_input('Height (cm)', 6.0, 40.0, 11.0, 0.5)
    dpi        = 600     if cli_mode else st.number_input('DPI', 72, 600, 300, 10)

    run = st.button('▶️ Run', type='primary')

if run:
    if uploaded is not None:
        df_raw = read_table(uploaded)
    elif path_text:
        if not Path(path_text).exists():
            st.error('Path not found.'); st.stop()
        df_raw = read_table(path_text)
    else:
        st.warning('Please upload a file or paste a path.'); st.stop()

    st.success(f'Loaded: {df_raw.shape[0]} taxa × {df_raw.shape[1]} samples')

    if not genera_list:
        st.warning('Genus list is empty.'); st.stop()
    df = filter_by_genus_cli_exact(df_raw, genera_list)
    if df.empty:
        st.warning('Empty after filtering. Ensure last-rank names match.'); st.stop()

    corr_df, p_df = spearman_corr_and_p(df, axis=axis_choice, fdr_alpha=fdr_alpha, method=p_adjust_method)
    mask = (p_df <= fdr_alpha)
    corr_df_masked = corr_df * mask
    corr_df_ord, p_df_ord = reorder_by_clustering_cli(corr_df_masked, p_df)

    matplotlib.rc('font', size=font_size)
    fig = plt.figure(figsize=(fig_w/2.54, fig_h/2.54), dpi=dpi)
    sns.heatmap(
        corr_df_ord, cmap=cmap, vmax=1, vmin=-1,
        xticklabels=True, yticklabels=True, annot=False,
        linewidths=linewidths, linecolor='white',
        cbar_kws={'label': 'Spearman correlation coefficient'}
    )
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader('📊 Correlation matrix (clustered order)'); st.dataframe(corr_df_ord)
    st.subheader('📊 Adjusted P-value matrix (same order)'); st.dataframe(p_df_ord)

    csv_corr = corr_df_ord.to_csv().encode('utf-8')
    csv_p    = p_df_ord.to_csv().encode('utf-8')
    st.download_button('⬇️ Download correlation CSV', data=csv_corr, file_name='microcoheat_corr.csv', mime='text/csv')
    st.download_button('⬇️ Download p-values CSV', data=csv_p, file_name='microcoheat_pvalues.csv', mime='text/csv')

    buf_png, buf_pdf = io.BytesIO(), io.BytesIO()
    fig = plt.figure(figsize=(fig_w/2.54, fig_h/2.54), dpi=dpi)
    sns.heatmap(
        corr_df_ord, cmap=cmap, vmax=1, vmin=-1,
        xticklabels=True, yticklabels=True, annot=False,
        linewidths=linewidths, linecolor='white',
        cbar_kws={'label': 'Spearman correlation coefficient'}
    )
    plt.tight_layout()
    fig.savefig(buf_png, format='png', dpi=dpi, bbox_inches='tight')
    fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
    buf_png.seek(0); buf_pdf.seek(0)
    st.download_button('🖼️ Download heatmap PNG', data=buf_png, file_name='microcoheat_heatmap.png', mime='image/png')
    st.download_button('📄 Download heatmap PDF', data=buf_pdf, file_name='microcoheat_heatmap.pdf', mime='application/pdf')
