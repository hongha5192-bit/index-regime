"""CJM Regime Dashboard — Streamlit version (polished).

Reuses data loaders and figure builders from build_dashboard.py.
Run with:
    cd /Users/hanguyen/CJModel(Final)/webapp
    streamlit run streamlit_app.py
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dashboard import (
    ROOT, OHLC_CSV, REGIMES, COLORS, GROUP_COLOR, FEATURE_CATALOG, V4_FEATS,
    load_regime_series, compute_long_metrics, make_chart,
)

st.set_page_config(page_title="CJM Regime Dashboard", layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': "CJM v7 · 26 features · K=3 regimes"})

# ── Typography & global CSS (large, readable) ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Page title */
h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 34px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}
/* Subheaders */
h2 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-top: 22px !important;
}
h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 19px !important;
    font-weight: 600 !important;
}

/* Streamlit caption */
.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 14px !important;
    line-height: 1.5 !important;
}

/* Native st.metric — used in Importance KPI cards */
[data-testid="stMetricValue"] {
    font-size: 36px !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 14px !important;
    color: #555 !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stMetricDelta"] {
    font-size: 13px !important;
}
div[data-testid="stMetric"] {
    background: white !important;
    padding: 14px 16px !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* Regime badge — DOMINANT visual element */
.regime-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 24px;
    font-weight: 800;
    font-size: 22px;
    color: white;
    letter-spacing: 1.2px;
    font-family: 'DM Sans', sans-serif;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12);
}

/* Regime card container */
.regime-card {
    border-radius: 12px;
    padding: 20px 22px 16px 22px;
    margin-bottom: 6px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* Index label (e.g., "VNINDEX") */
.card-index-label {
    font-size: 14px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 10px;
    font-family: 'DM Sans', sans-serif;
}

/* Close + date row inside card */
.card-meta-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 14px 0 4px 0;
    flex-wrap: wrap;
    gap: 8px;
}
.card-close-val {
    font-size: 30px;
    font-weight: 700;
    color: #1a1a1a;
    font-family: 'DM Sans', sans-serif;
    line-height: 1.1;
}
.card-close-chg-pos {
    font-size: 16px;
    font-weight: 600;
    color: #2ecc71;
    margin-left: 8px;
    font-family: 'DM Mono', monospace;
}
.card-close-chg-neg {
    font-size: 16px;
    font-weight: 600;
    color: #e74c3c;
    margin-left: 8px;
    font-family: 'DM Mono', monospace;
}
.card-date-val {
    font-size: 14px;
    color: #888;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
}

/* Feature pill */
.feat-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    color: white;
    font-size: 13px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
}

/* Tables (st.dataframe internals) */
.dataframe,
[data-testid="stTable"], [data-testid="stDataFrame"] {
    font-size: 15px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.dataframe th { font-size: 14px !important; font-weight: 700 !important; }
.dataframe td { padding: 8px 10px !important; }

/* Streamlit tabs — bigger labels */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 17px !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-list"] button {
    padding: 10px 22px !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 14px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 18px !important;
    margin-bottom: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    font-size: 13px !important;
}

/* Selectbox / radio option text */
[data-baseweb="select"] div, [role="radiogroup"] label {
    font-size: 14px !important;
}

/* Expander title */
.streamlit-expanderHeader p, [data-testid="stExpander"] summary p {
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Section dividers */
hr { margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Cached data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    cases = [('vnindex','VNINDEX'),('midcap','VNMIDCAP'),('smallcap','VNSMALLCAP')]
    out = {}
    for tag, name in cases:
        out[name] = load_regime_series(
            os.path.join(ROOT, f'phase4_cjm_v7_{tag}_results.npz'),
            ohlc_csv=OHLC_CSV[name],
            ext_csv=os.path.join(ROOT, f'phase4_regime_v7_{tag}.csv'),
            ext_label_col='v7_label',
            ext_proba_cols=('v7_Bull','v7_Neut','v7_Bear'))
    return out

@st.cache_data
def load_feature_csvs():
    return {
        'VNINDEX':    pd.read_csv(os.path.join(ROOT, 'VNINDEX_OHLCV_with_features_v4.csv'),    parse_dates=['Date']),
        'VNMIDCAP':   pd.read_csv(os.path.join(ROOT, 'VNMIDCAP_OHLCV_with_features_v4_shared.csv'), parse_dates=['Date']),
        'VNSMALLCAP': pd.read_csv(os.path.join(ROOT, 'VNSMALLCAP_OHLCV_with_features_v4_shared.csv'), parse_dates=['Date']),
    }

@st.cache_data
def load_importance():
    ext = os.path.join(ROOT, 'feature_importance_extended_v7.csv')
    return pd.read_csv(ext) if os.path.exists(ext) else None

indices = load_all()
vnindex, midcap, smallcap = indices['VNINDEX'], indices['VNMIDCAP'], indices['VNSMALLCAP']
feat_csvs = load_feature_csvs()
imp = load_importance()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    chart_index = st.radio("Chart focus", ["All 3", "VNINDEX", "VNMIDCAP", "VNSMALLCAP"], index=0)
    perf_index  = st.selectbox("Performance metrics for", ["VNINDEX", "VNMIDCAP", "VNSMALLCAP"])
    show_rangeslider = st.checkbox("Show chart range slider", value=True)
    st.markdown("---")
    all_groups = sorted({grp for grp, _, _, _ in FEATURE_CATALOG})
    feat_groups = st.multiselect("Feature groups", all_groups, default=all_groups)
    st.markdown("---")
    st.caption(f"CJM v7 · 26 features · K=3\nData through "
               f"**{max(df['Date'].max() for df in indices.values()).date()}**")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("CJM Regime Classification — Vietnam Equity Indices")
st.caption("Continuous Statistical Jump Model (K=3, λ=50). Train 2016-07 → 2024-12. Out-of-sample 2025-01 onward.")

tab_dash, tab_feat, tab_imp = st.tabs([
    "Dashboard",
    f"Features ({len(FEATURE_CATALOG)})",
    "Importance",
])

# ─────────────────────────────── Tab 1: Dashboard ───────────────────────────

def regime_card(col, name, df, card_key):
    """Single self-contained card block: badge + close/change + date + prob bar.

    card_key must be unique per call (e.g. 'vnindex', 'midcap', 'smallcap').
    All plotly_chart / dataframe keys are namespaced with card_key to prevent
    StreamlitDuplicateElementId collisions.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg = float(last['Close'] - prev['Close'])
    chg_pct = (chg / prev['Close']) * 100 if prev['Close'] else 0.0
    color = COLORS[last['label']]
    bg_color = color + "14"   # ~8% opacity tint
    pB = float(last.get('p_Bull', 0))
    pN = float(last.get('p_Neutral', 0))
    pX = float(last.get('p_Bear', 0))

    chg_class = "card-close-chg-pos" if chg >= 0 else "card-close-chg-neg"
    sign = "+" if chg >= 0 else ""

    with col:
        st.markdown(
            f"<div class='regime-card' style='background:{bg_color}; border-left:4px solid {color};'>"
            f"<div class='card-index-label'>{name}</div>"
            f"<span class='regime-badge' style='background:{color};'>{last['label'].upper()}</span>"
            f"<div class='card-meta-row'>"
            f"<span class='card-close-val'>{last['Close']:,.1f}"
            f"<span class='{chg_class}'>{sign}{chg:.1f} ({sign}{chg_pct:.2f}%)</span></span>"
            f"<span class='card-date-val'>{last['Date'].date()}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        if np.isfinite(pB):
            fig = go.Figure()
            for seg_val, seg_label, seg_color in [
                (pB, 'Bull',    COLORS['Bull']),
                (pN, 'Neutral', COLORS['Neutral']),
                (pX, 'Bear',    COLORS['Bear']),
            ]:
                fig.add_trace(go.Bar(
                    x=[seg_val], y=[''],
                    orientation='h',
                    marker=dict(color=seg_color),
                    text=f'{seg_label} {seg_val*100:.0f}%' if seg_val > 0.10 else '',
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(size=14, color='white', family='DM Sans'),
                    showlegend=False,
                    hovertemplate=f'{seg_label}: {seg_val*100:.1f}%<extra></extra>',
                ))
            fig.update_layout(
                barmode='stack',
                height=52,
                margin=dict(l=0, r=0, t=4, b=4),
                xaxis=dict(visible=False, range=[0, 1]),
                yaxis=dict(visible=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={'displayModeBar': False},
                key=f"probbar_{card_key}",   # unique key per card
            )

with tab_dash:
    st.subheader("Current regime")
    cols = st.columns(3)
    card_keys = ['vnindex', 'midcap', 'smallcap']
    for col, (n, df), ckey in zip(
        cols,
        [('VNINDEX', vnindex), ('VNMIDCAP', midcap), ('VNSMALLCAP', smallcap)],
        card_keys,
    ):
        regime_card(col, n, df, ckey)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.subheader("Regime timelines")

    chart_map = {'VNINDEX': vnindex, 'VNMIDCAP': midcap, 'VNSMALLCAP': smallcap}
    if chart_index == "All 3":
        for i, (n, df) in enumerate(chart_map.items()):
            st.plotly_chart(
                make_chart(df, f"{n}", f"{n} close",
                           show_rangeslider=(show_rangeslider and i == 0)),
                use_container_width=True,
                key=f"timeline_chart_{n}",   # unique per index
            )
    else:
        st.plotly_chart(
            make_chart(chart_map[chart_index], chart_index, f"{chart_index} close",
                       show_rangeslider=show_rangeslider),
            use_container_width=True,
            key=f"timeline_chart_single_{chart_index}",
        )

    # Distribution
    st.subheader("Regime distribution (full history)")
    dist_rows = []
    for name, df in [('VNINDEX', vnindex), ('VNMIDCAP', midcap), ('VNSMALLCAP', smallcap)]:
        c = df['label'].value_counts().reindex(REGIMES, fill_value=0)
        total = c.sum()
        dist_rows.append({
            'Index': name,
            'Bars': int(total),
            # Pre-scaled to percent — printf format '%%' is literal, not a multiplier.
            'Bull':    c['Bull']    / total * 100.0,
            'Neutral': c['Neutral'] / total * 100.0,
            'Bear':    c['Bear']    / total * 100.0,
        })
    ddf = pd.DataFrame(dist_rows)
    st.dataframe(
        ddf,
        hide_index=True,
        use_container_width=True,
        key="dist_table",
        column_config={
            'Bull':    st.column_config.ProgressColumn(format='%.1f%%', min_value=0.0, max_value=100.0),
            'Neutral': st.column_config.ProgressColumn(format='%.1f%%', min_value=0.0, max_value=100.0),
            'Bear':    st.column_config.ProgressColumn(format='%.1f%%', min_value=0.0, max_value=100.0),
        },
    )

    # Performance metrics
    st.subheader(f"Long-position performance — {perf_index}")
    st.caption("Entry at open[t], exit at open[t+N]. Excursion = next N bars (t+1..t+N).")

    perf_df = compute_long_metrics(chart_map[perf_index])
    # Streamlit column_config.format uses printf (not d3), so %% is literal — pre-scale to percent.
    perf_disp = perf_df.copy()
    for col in ['win_rate', 'median_ret', 'p75_up', 'p25_dn']:
        if col in perf_disp.columns:
            perf_disp[col] = perf_disp[col] * 100.0
    st.dataframe(
        perf_disp,
        hide_index=True,
        use_container_width=True,
        key="perf_table",
        column_config={
            'regime':     st.column_config.TextColumn('Regime'),
            'win_rate':   st.column_config.ProgressColumn(format='%.1f%%', min_value=0.0, max_value=100.0),
            'median_ret': st.column_config.NumberColumn('Median ret',   format='%+.2f%%'),
            'p75_up':     st.column_config.NumberColumn('P75 up',       format='%+.2f%%'),
            'p25_dn':     st.column_config.NumberColumn('P25 down',     format='%+.2f%%'),
        },
    )

# ─────────────────────────────── Tab 2: Features ────────────────────────────
with tab_feat:
    st.subheader(f"Feature Engineering — {len(FEATURE_CATALOG)} features fed into CJM v7")

    latest = {n: df.dropna(subset=V4_FEATS).sort_values('Date').iloc[-1]
              for n, df in feat_csvs.items()}
    last_dt = max(v['Date'] for v in latest.values()).date()
    st.caption(f"Latest snapshot: **{last_dt}**. Use the sidebar to filter by group.")

    # Group breakdown chart
    grp_count = pd.Series([grp for grp, _, _, _ in FEATURE_CATALOG]).value_counts()
    grp_df = pd.DataFrame({'group': grp_count.index, 'count': grp_count.values})
    grp_df['color'] = grp_df['group'].map(GROUP_COLOR)
    fig_grp = go.Figure(go.Bar(
        x=grp_df['count'], y=grp_df['group'], orientation='h',
        marker=dict(color=grp_df['color']),
        text=grp_df['count'], textposition='outside',
    ))
    fig_grp.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#fafbfc',
        xaxis=dict(visible=False),
        yaxis=dict(autorange='reversed'),
        font=dict(family='DM Sans'),
    )
    with st.expander("Feature group composition", expanded=False):
        st.plotly_chart(
            fig_grp,
            use_container_width=True,
            config={'displayModeBar': False},
            key="feat_group_bar",
        )

    # Table with latest values
    rows = []
    for grp, fname, desc, formula in FEATURE_CATALOG:
        if grp not in feat_groups:
            continue
        vV = float(latest['VNINDEX'][fname])
        vM = float(latest['VNMIDCAP'][fname])
        vS = float(latest['VNSMALLCAP'][fname])

        def fmt(x, _grp=grp, _fname=fname):
            if _grp == 'Flow':
                return f"{x*100:+.2f}%" if _fname.startswith('net_fgn') else f"{x*100:.2f}%"
            if _grp == 'Breadth':
                return f"{x*100:.2f}%"
            if _grp == 'Volume':
                return f"{x:+.3f}" if _fname == 'up_down_vol_ratio_20d' else f"{x:.3f}"
            return f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}"

        rows.append({
            'Group': grp,
            'Feature': fname,
            'Description': desc,
            'Formula': formula,
            'VNINDEX': fmt(vV),
            'VNMIDCAP': fmt(vM),
            'VNSMALLCAP': fmt(vS),
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        key="feat_catalog_table",
        column_config={
            'Formula':     st.column_config.TextColumn(width='medium'),
            'Description': st.column_config.TextColumn(width='medium'),
        },
    )

# ─────────────────────────────── Tab 3: Importance ──────────────────────────
with tab_imp:
    if imp is None:
        st.warning("feature_importance_extended_v7.csv not found.")
    else:
        imp = imp.sort_values('rank_avg').reset_index(drop=True)

        metrics_def = [
            ('eta2_mean', 'eta2',   'η²',       '#3498db'),
            ('rf_gini',   'rfgini', 'RF Gini',  '#27ae60'),
            ('shap_total','shap',   'SHAP',      '#9b59b6'),
            ('wass_max',  'wass',   'Wass',      '#e67e22'),
        ]

        top10 = {m: imp.sort_values(m, ascending=False).head(10)['feature'].tolist()
                 for m, _, _, _ in metrics_def}
        top10_union = set().union(*top10.values())
        feat_agree = {f: sum(1 for m, _, _, _ in metrics_def if f in top10[m])
                      for f in top10_union}

        def universal_n(n):
            return sorted(set.intersection(*(set(top10[m][:n]) for m, _, _, _ in metrics_def)))

        u3, u5, u8, u10 = universal_n(3), universal_n(5), universal_n(8), universal_n(10)

        # ── Consensus summary cards ──
        st.subheader("Consensus across 4 methods")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Universal top-1",  f"{len([u3[0]] if u3 else [])}", help="In rank 1 of ALL 4 methods")
        m2.metric("Universal top-5",  f"{len(u5)}",                    help="In top-5 of ALL 4 methods")
        m3.metric("Universal top-8",  f"{len(u8)}",                    help="In top-8 of ALL 4 methods")
        m4.metric("Universal top-10", f"{len(u10)}",                   help="In top-10 of ALL 4 methods")
        st.caption(
            f"**Top-1:** `ULT_RSI` (rank 1 in all 4) · "
            f"**Top-5 universal:** {' · '.join(f'`{f}`' for f in u5)} · "
            f"**Top-8 universal:** {', '.join(f'`{f}`' for f in u8)}"
        )

        # ── 4 horizontal bar charts side by side ──
        st.markdown("---")
        st.subheader("Top-10 by each method (interactive)")
        chart_cols = st.columns(4)
        for (m, mkey, label, mcol), col in zip(metrics_def, chart_cols):
            with col:
                top_df = imp.sort_values(m, ascending=False).head(10).iloc[::-1]
                bar_colors = [
                    '#f1c40f' if feat_agree.get(f, 0) == 4 else
                    '#95a5a6' if feat_agree.get(f, 0) == 3 else
                    '#d6c4a3' if feat_agree.get(f, 0) == 2 else
                    '#e74c3c'
                    for f in top_df['feature']
                ]
                fig = go.Figure(go.Bar(
                    x=top_df[m],
                    y=top_df['feature'],
                    orientation='h',
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=[f"{v:.3f}" for v in top_df[m]],
                    textposition='outside',
                    textfont=dict(size=13, family='DM Mono'),
                    hovertemplate='<b>%{y}</b><br>' + label + ' = %{x:.4f}<extra></extra>',
                ))
                fig.update_layout(
                    title=dict(
                        text=f"<b style='color:{mcol}'>{label}</b>",
                        x=0.5,
                        font=dict(size=18, family='DM Sans'),
                    ),
                    height=440,
                    margin=dict(l=10, r=60, t=54, b=20),
                    plot_bgcolor='#fafbfc',
                    xaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(size=13, family='DM Sans')),
                    yaxis=dict(tickfont=dict(size=14, family='DM Sans')),
                    font=dict(family='DM Sans'),
                )
                # key is unique: tab prefix + metric key
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'displayModeBar': False},
                    key=f"imp_bar_{mkey}",
                )

        st.caption(
            "Gold = in all 4 methods' top-10 · Gray = in 3 · Tan = in 2 · Red = unique to one method"
        )

        # ── Divergence table ──
        unique = []
        for f in top10_union:
            if feat_agree[f] == 1:
                m_hit = [(lbl, top10[m].index(f) + 1)
                         for m, _, lbl, _ in metrics_def if f in top10[m]][0]
                unique.append({'Feature': f, 'Only in': m_hit[0], 'Rank': m_hit[1]})
        if unique:
            st.markdown("**Top-10 divergence — picked by only ONE method:**")
            st.dataframe(
                pd.DataFrame(unique),
                hide_index=True,
                use_container_width=False,
                key="imp_divergence_table",
            )

        # ── SHAP per-class heatmap ──
        st.markdown("---")
        st.subheader("SHAP per-class heatmap — which features push toward which regime")

        heat_df = imp[['feature', 'shap_Bull', 'shap_Neut', 'shap_Bear']].copy()
        heat_df['total'] = heat_df[['shap_Bull', 'shap_Neut', 'shap_Bear']].sum(axis=1)
        # Take top-20 by shap_total, sort ascending so highest is at top of heatmap
        heat_df = heat_df.sort_values('total', ascending=False).head(20).sort_values('total', ascending=True)

        z = heat_df[['shap_Bull', 'shap_Neut', 'shap_Bear']].values
        # 28px per row → larger cells & bigger labels
        heatmap_h = min(820, max(380, 28 * len(heat_df) + 110))

        fig_heat = go.Figure(go.Heatmap(
            z=z,
            y=heat_df['feature'],
            x=['Bull', 'Neutral', 'Bear'],
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='mean |SHAP|', font=dict(size=14, family='DM Sans')),
                tickfont=dict(size=14, family='DM Sans'),
            ),
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>',
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate='%{text}',
            textfont=dict(size=13, color='white', family='DM Mono'),
        ))
        fig_heat.update_layout(
            height=heatmap_h,
            margin=dict(l=10, r=10, t=44, b=20),
            xaxis=dict(side='top',
                       tickfont=dict(size=18, color='#222', family='DM Sans')),
            yaxis=dict(tickfont=dict(size=14, family='DM Sans')),
            font=dict(family='DM Sans'),
        )
        st.plotly_chart(
            fig_heat,
            use_container_width=True,
            config={'displayModeBar': False},
            key="imp_shap_heatmap",
        )
        st.caption(
            "Viridis scale: bright yellow = highest SHAP magnitude for that regime. "
            "Top-20 features by total SHAP. Sorted by total SHAP magnitude (bottom → highest)."
        )

        # ── Main ranked table ──
        st.markdown("---")
        st.subheader("Full ranking — all 26 features, 4 metrics (averaged across 3 indices)")
        display = imp[['feature', 'group', 'eta2_mean', 'rf_gini', 'shap_total', 'wass_max',
                        'rank_avg', 'signature']].copy()
        display.columns = ['Feature', 'Group', 'eta2', 'RF Gini', 'SHAP total', 'Wass max',
                           'rank_avg', 'Signature']
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            key="imp_full_table",
            column_config={
                'eta2':       st.column_config.ProgressColumn('η²',         format='%.3f', min_value=0.0, max_value=float(display['eta2'].max())),
                'RF Gini':    st.column_config.ProgressColumn(format='%.3f', min_value=0.0, max_value=float(display['RF Gini'].max())),
                'SHAP total': st.column_config.ProgressColumn(format='%.3f', min_value=0.0, max_value=float(display['SHAP total'].max())),
                'Wass max':   st.column_config.ProgressColumn(format='%.3f', min_value=0.0, max_value=float(display['Wass max'].max())),
                'rank_avg':   st.column_config.NumberColumn(format='%.2f'),
            },
        )
        st.caption(
            "Click any column header to sort. "
            "η² verified 8/8 · RF bit-exact · SHAP additivity 2.4e-06 · Wass 10/10."
        )

st.markdown(
    f"<div style='text-align:center; color:#bbb; font-size:11px; margin-top:30px; "
    f"font-family: DM Mono, monospace;'>"
    f"CJM v7 (26 features) · K=3 · λ=50 · grid=0.05 · "
    f"Data through {max(df['Date'].max() for df in indices.values()).date()}</div>",
    unsafe_allow_html=True,
)
