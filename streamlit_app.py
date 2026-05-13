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
    load_regime_series, compute_long_metrics, make_chart, regime_segments,
)

TRAIN_END = pd.Timestamp('2025-01-01')


def _run_lengths_by_label(df, mask, global_last_date=None):
    """Returns dict label → list of run lengths (in bars) within rows selected by mask.

    Excludes the in-progress run (segment whose end equals global_last_date).
    """
    sub = df.loc[mask].reset_index(drop=True)
    segs = regime_segments(sub)
    out = {lab: [] for lab in REGIMES}
    for start, end, lab in segs:
        if global_last_date is not None and end == global_last_date:
            continue
        length = int(((sub['Date'] >= start) & (sub['Date'] <= end)).sum())
        out[lab].append(length)
    return out


def _run_magnitudes_by_label(df, mask, global_last_date=None):
    """Returns dict label → list of % returns from lowest close in run → end-of-run close.

    Excludes the in-progress run: any segment whose end date equals the global
    last date is dropped, since its low→close return is not yet final.
    """
    sub = df.loc[mask].reset_index(drop=True)
    segs = regime_segments(sub)
    out = {lab: [] for lab in REGIMES}
    for start, end, lab in segs:
        if global_last_date is not None and end == global_last_date:
            continue
        run = sub[(sub['Date'] >= start) & (sub['Date'] <= end)]
        if len(run) == 0:
            continue
        low = run['Close'].min()
        e = run['Close'].iloc[-1]
        if low and np.isfinite(low) and np.isfinite(e):
            out[lab].append((e / low - 1.0) * 100.0)
    return out


def compute_summary_stats(df):
    """Compute run-length & magnitude distributions per regime, in train vs test.

    The in-progress run (last segment ending on the global last date) is
    excluded from the historical pool so the displayed median/P75 reflects
    only completed runs.
    """
    df = df.sort_values('Date').reset_index(drop=True)
    train_mask = df['Date'] < TRAIN_END
    test_mask  = df['Date'] >= TRAIN_END
    last_date = df['Date'].iloc[-1]
    return {
        'train_runs': _run_lengths_by_label(df, train_mask, global_last_date=last_date),
        'test_runs':  _run_lengths_by_label(df, test_mask,  global_last_date=last_date),
        'train_mags': _run_magnitudes_by_label(df, train_mask, global_last_date=last_date),
        'test_mags':  _run_magnitudes_by_label(df, test_mask,  global_last_date=last_date),
    }


def current_run_info(df):
    """Returns (run_start_date, run_label, run_length_bars, pct_from_run_low,
    run_low_date, run_low_close)."""
    segs = regime_segments(df)
    if not segs:
        return None, None, 0, 0.0, None, None
    start, end, lab = segs[-1]
    sub = df[(df['Date'] >= start) & (df['Date'] <= end)].reset_index(drop=True)
    length = len(sub)
    low_idx = sub['Close'].idxmin()
    low_date = sub.loc[low_idx, 'Date']
    low_close = sub.loc[low_idx, 'Close']
    last_close = sub['Close'].iloc[-1]
    pct = (last_close / low_close - 1.0) * 100.0 if low_close else 0.0
    return start, lab, length, pct, low_date, low_close


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: (len(df), df['Date'].iloc[-1])})
def forward_stats_cached(df):
    """Return compute_long_metrics filtered to (3,5) for both splits. Cached per-df by length+last-date."""
    return compute_long_metrics(df, periods=(3, 5))


def forward_stats_for_regime(df, regime_label):
    """Returns dict {(split, period): {n, win_rate, median_ret, p75_up, p25_dn}}."""
    agg = forward_stats_cached(df)
    sub = agg[agg['regime'] == regime_label]
    out = {}
    for _, r in sub.iterrows():
        out[(r['split'], r['period'])] = {
            'n': int(r['n']),
            'win': float(r['win_rate']),
            'med': float(r['median_ret']),
            'up':  float(r['p75_up']),
            'dn':  float(r['p25_dn']),
        }
    return out


def trough_3m(df, n_bars=63):
    """Lowest Close in last n_bars bars and its date; plus % above trough."""
    sub = df.tail(n_bars)
    if len(sub) == 0:
        return None, None, 0.0
    idx = sub['Close'].idxmin()
    low = sub.loc[idx, 'Close']
    low_date = sub.loc[idx, 'Date']
    last = df['Close'].iloc[-1]
    pct_above = (last / low - 1.0) * 100.0 if low else 0.0
    return low_date, low, pct_above

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
# Cache key derived from data-file mtimes so any deploy with refreshed CSVs
# busts the cache automatically (otherwise stale data can survive a restart).
def _data_mtime_key():
    paths = [
        os.path.join(ROOT, 'phase4_cjm_v7_vnindex_results.npz'),
        os.path.join(ROOT, 'phase4_cjm_v7_midcap_results.npz'),
        os.path.join(ROOT, 'phase4_cjm_v7_smallcap_results.npz'),
        os.path.join(ROOT, 'phase4_regime_v7_vnindex.csv'),
        os.path.join(ROOT, 'phase4_regime_v7_midcap.csv'),
        os.path.join(ROOT, 'phase4_regime_v7_smallcap.csv'),
        os.path.join(ROOT, 'VNINDEX_OHLCV_with_features_v4.csv'),
        os.path.join(ROOT, 'VNMIDCAP_OHLCV_with_features_v4_shared.csv'),
        os.path.join(ROOT, 'VNSMALLCAP_OHLCV_with_features_v4_shared.csv'),
        os.path.join(ROOT, 'feature_importance_extended_v7.csv'),
    ]
    return tuple(int(os.path.getmtime(p)) for p in paths if os.path.exists(p))


@st.cache_data
def load_all(_mtime_key):
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
def load_feature_csvs(_mtime_key):
    return {
        'VNINDEX':    pd.read_csv(os.path.join(ROOT, 'VNINDEX_OHLCV_with_features_v4.csv'),    parse_dates=['Date']),
        'VNMIDCAP':   pd.read_csv(os.path.join(ROOT, 'VNMIDCAP_OHLCV_with_features_v4_shared.csv'), parse_dates=['Date']),
        'VNSMALLCAP': pd.read_csv(os.path.join(ROOT, 'VNSMALLCAP_OHLCV_with_features_v4_shared.csv'), parse_dates=['Date']),
    }

@st.cache_data
def load_importance(_mtime_key):
    ext = os.path.join(ROOT, 'feature_importance_extended_v7.csv')
    return pd.read_csv(ext) if os.path.exists(ext) else None

_mtime = _data_mtime_key()
indices = load_all(_mtime)
vnindex, midcap, smallcap = indices['VNINDEX'], indices['VNMIDCAP'], indices['VNSMALLCAP']
feat_csvs = load_feature_csvs(_mtime)
imp = load_importance(_mtime)

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
_latest_data = max(df['Date'].max() for df in indices.values()).date()
st.title("CJM Regime Classification — Vietnam Equity Indices")
st.markdown(
    f"<div style='display:flex; align-items:center; gap:12px; margin-top:-6px; margin-bottom:8px;'>"
    f"<span style='color:#666; font-size:13px;'>Continuous Statistical Jump Model (K=3, λ=50). "
    f"Train 2016-07 → 2024-12. Out-of-sample 2025-01 onward.</span>"
    f"<span style='display:inline-block; padding:3px 10px; background:#27ae60; color:#fff; "
    f"font-size:11px; font-weight:700; border-radius:12px; letter-spacing:0.5px;'>"
    f"DATA THROUGH {_latest_data}</span></div>",
    unsafe_allow_html=True,
)

tab_summary, tab_dash, tab_feat, tab_imp, tab_lm = st.tabs([
    "Summary",
    "Dashboard",
    f"Features ({len(FEATURE_CATALOG)})",
    "Feature Importance",
    "Stock Ranking",
])

# ─────────────────────────────── Tab 0: Summary ─────────────────────────────

def _delta_chip(label, today, prev, color):
    """Render a small chip showing today's prob + delta vs prev day."""
    delta = today - prev
    if abs(delta) < 0.005:
        arrow = "→"; dcolor = "#888"
    elif delta > 0:
        arrow = "▲"; dcolor = "#e67e22" if label != 'Bull' else "#27ae60"
    else:
        arrow = "▼"; dcolor = "#888"
    # Highlight any non-dominant non-zero prob (even small) with bold
    bg_intensity = max(0.06, min(today, 1.0) * 0.40)
    bg = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{bg_intensity:.2f})"
    weight = '700' if today >= 0.05 else '500'
    return (
        f"<div style='display:inline-block; padding:6px 12px; margin-right:6px; "
        f"border-radius:8px; background:{bg}; border:1px solid {color}55; min-width:100px;'>"
        f"<div style='font-size:10px; font-weight:600; color:{color}; letter-spacing:0.5px;'>{label.upper()}</div>"
        f"<div style='font-size:18px; font-weight:{weight}; color:#222; font-variant-numeric:tabular-nums;'>"
        f"{today*100:.0f}%"
        f"<span style='font-size:11px; color:{dcolor}; margin-left:6px; font-weight:600;'>"
        f"{arrow} {delta*100:+.0f}pp</span></div></div>"
    )


def summary_card(col, name, df, stats):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    label = last['label']
    color = COLORS[label]
    chg = float(last['Close'] - prev['Close'])
    chg_pct = (chg / prev['Close']) * 100 if prev['Close'] else 0.0
    sign = "+" if chg >= 0 else ""

    pB_t, pN_t, pX_t = float(last['p_Bull']), float(last['p_Neutral']), float(last['p_Bear'])
    pB_y, pN_y, pX_y = float(prev['p_Bull']), float(prev['p_Neutral']), float(prev['p_Bear'])

    # Q3 — duration
    run_start, run_label, run_len, run_pct, run_low_date, run_low_close = current_run_info(df)
    train_runs = stats['train_runs'].get(run_label, [])
    test_runs  = stats['test_runs'].get(run_label, [])
    train_med  = int(np.median(train_runs)) if train_runs else 0
    train_p75  = int(np.percentile(train_runs, 75)) if train_runs else 0
    test_med   = int(np.median(test_runs)) if test_runs else 0
    pct_rank = (np.sum(np.array(train_runs + test_runs) <= run_len) / max(1, len(train_runs)+len(test_runs))) * 100

    # Q4 — magnitude
    low_date, low_val, pct_above_low = trough_3m(df)
    train_mags = stats['train_mags'].get(run_label, [])
    test_mags  = stats['test_mags'].get(run_label, [])
    all_mags   = train_mags + test_mags
    mag_med    = float(np.median(all_mags)) if all_mags else 0.0
    mag_p75    = float(np.percentile(all_mags, 75)) if all_mags else 0.0

    # Shift detection — any non-dominant prob ≥ 5% is a yellow flag
    shifts = []
    for lab, p, p_prev in [('Bull', pB_t, pB_y), ('Neutral', pN_t, pN_y), ('Bear', pX_t, pX_y)]:
        if lab == label:
            continue
        if p >= 0.05 or (p - p_prev) >= 0.05:
            shifts.append(f"P({lab}) = {p*100:.0f}% (was {p_prev*100:.0f}%)")
    if shifts:
        shift_html = (
            f"<div style='margin-top:10px; padding:10px 14px; background:#fff8e1; "
            f"border-left:4px solid #f39c12; border-radius:6px;'>"
            f"<div style='font-size:11px; font-weight:700; color:#b9770e; letter-spacing:0.5px;'>"
            f"⚠ REGIME SHIFT SIGNAL</div>"
            f"<div style='font-size:13px; color:#222; margin-top:3px;'>" + " · ".join(shifts) + "</div></div>"
        )
    else:
        shift_html = (
            f"<div style='margin-top:10px; padding:10px 14px; background:#eafaf1; "
            f"border-left:4px solid #2ecc71; border-radius:6px;'>"
            f"<div style='font-size:11px; font-weight:700; color:#1e8449; letter-spacing:0.5px;'>"
            f"✓ STABLE — no rival regime probability above 5%</div></div>"
        )

    with col:
        # Q1 — Latest regime
        st.markdown(
            f"<div style='padding:16px 18px; background:{color}14; border-left:5px solid {color}; "
            f"border-radius:10px; margin-bottom:14px;'>"
            f"<div style='font-size:13px; font-weight:600; color:#666; letter-spacing:1px;'>{name}</div>"
            f"<div style='display:flex; align-items:baseline; gap:14px; margin-top:6px;'>"
            f"<span style='font-size:30px; font-weight:800; color:{color}; letter-spacing:0.5px;'>{label.upper()}</span>"
            f"<span style='font-size:18px; font-weight:700; color:#222;'>{last['Close']:,.1f}</span>"
            f"<span style='font-size:14px; font-weight:600; "
            f"color:{'#27ae60' if chg>=0 else '#c0392b'};'>{sign}{chg:.1f} ({sign}{chg_pct:.2f}%)</span>"
            f"</div>"
            f"<div style='font-size:11px; color:#888; margin-top:4px;'>as of {last['Date'].date()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Q2 — Probability vector with day-over-day delta
        st.markdown("<div style='font-size:12px; font-weight:700; color:#666; "
                    "letter-spacing:1px; margin:6px 0 8px 0;'>REGIME PROBABILITIES (vs yesterday)</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div>" +
            _delta_chip('Bull',    pB_t, pB_y, COLORS['Bull']) +
            _delta_chip('Neutral', pN_t, pN_y, COLORS['Neutral']) +
            _delta_chip('Bear',    pX_t, pX_y, COLORS['Bear']) +
            "</div>" + shift_html,
            unsafe_allow_html=True,
        )

        # Q3 — Duration
        st.markdown(
            f"<div style='margin-top:14px; padding:12px 14px; background:#f7f8fa; border-radius:8px;'>"
            f"<div style='font-size:11px; font-weight:700; color:#666; letter-spacing:1px;'>DURATION IN {run_label.upper() if run_label else 'REGIME'}</div>"
            f"<div style='font-size:26px; font-weight:800; color:#222; margin-top:4px; font-variant-numeric:tabular-nums;'>"
            f"{run_len} <span style='font-size:14px; font-weight:500; color:#888;'>bars</span></div>"
            f"<div style='font-size:12px; color:#555; margin-top:4px;'>"
            f"since <b>{pd.Timestamp(run_start).date()}</b> · "
            f"train median {train_med} · train P75 {train_p75} · test median {test_med}"
            f"</div>"
            f"<div style='font-size:11px; color:{'#e67e22' if pct_rank>=75 else '#888'}; margin-top:3px; font-weight:600;'>"
            f"≤ this length in {pct_rank:.0f}% of historical {run_label} runs"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Q5 — Forward stats conditional on current regime (T+3, T+5)
        fwd = forward_stats_for_regime(df, label)
        def cell(split, period):
            v = fwd.get((split, period))
            if v is None:
                return "<td colspan='4' style='color:#aaa;'>no data</td>"
            ret_color = '#1e7e34' if v['med'] >= 0 else '#c0392b'
            return (
                f"<td style='text-align:center; font-weight:700;'>{v['win']*100:.0f}%</td>"
                f"<td style='text-align:center; font-weight:700; color:{ret_color};'>"
                f"{v['med']*100:+.2f}%</td>"
                f"<td style='text-align:center; color:#1e7e34;'>+{v['up']*100:.1f}%</td>"
                f"<td style='text-align:center; color:#c0392b;'>{v['dn']*100:.1f}%</td>"
            )
        n_train = fwd.get(('Train (≤2024)','T+3'), {}).get('n', 0)
        n_test  = fwd.get(('Test (2025+)','T+3'), {}).get('n', 0)
        st.markdown(
            f"<div style='margin-top:10px; padding:12px 14px; background:#fafbfc; "
            f"border:1px solid #e8eaee; border-radius:8px;'>"
            f"<div style='font-size:11px; font-weight:700; color:#666; letter-spacing:1px;'>"
            f"FORWARD STATS · GIVEN {label.upper()}</div>"
            f"<table style='width:100%; border-collapse:collapse; margin-top:6px; "
            f"font-size:11px; font-variant-numeric:tabular-nums;'>"
            f"<thead><tr style='color:#888; font-weight:600;'>"
            f"<th style='text-align:left; padding:2px 4px;'></th>"
            f"<th style='text-align:center; padding:2px 4px;'>Win</th>"
            f"<th style='text-align:center; padding:2px 4px;'>Median</th>"
            f"<th style='text-align:center; padding:2px 4px;'>P75 up</th>"
            f"<th style='text-align:center; padding:2px 4px;'>P25 dn</th></tr></thead>"
            f"<tbody>"
            f"<tr><td style='color:#444; font-weight:600; padding:3px 4px;'>Train T+3</td>"
            + cell('Train (≤2024)','T+3') + "</tr>"
            f"<tr><td style='color:#444; font-weight:600; padding:3px 4px;'>Train T+5</td>"
            + cell('Train (≤2024)','T+5') + "</tr>"
            f"<tr><td style='color:#444; font-weight:600; padding:3px 4px;'>Test T+3</td>"
            + cell('Test (2025+)','T+3') + "</tr>"
            f"<tr><td style='color:#444; font-weight:600; padding:3px 4px;'>Test T+5</td>"
            + cell('Test (2025+)','T+5') + "</tr>"
            f"</tbody></table>"
            f"<div style='font-size:10px; color:#999; margin-top:4px;'>"
            f"n_train={n_train} · n_test={n_test} bars classified as {label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Q4 — Magnitude. ONE primary metric (run progress, low→close), with verdict
        # pill vs historical completed runs. 3M low demoted to a footnote.
        if run_pct >= mag_p75:
            verdict_txt, verdict_bg, verdict_fg = "Above P75", "#27ae60", "#fff"
        elif run_pct >= mag_med:
            verdict_txt, verdict_bg, verdict_fg = "Above median", "#a3d977", "#1e7e34"
        elif run_pct >= 0:
            verdict_txt, verdict_bg, verdict_fg = "Below median", "#f4d35e", "#7a5c00"
        else:
            verdict_txt, verdict_bg, verdict_fg = "Negative", "#e74c3c", "#fff"

        st.markdown(
            f"<div style='margin-top:10px; padding:14px 16px; background:#f7f8fa; border-radius:8px;'>"
            # Header
            f"<div style='font-size:11px; font-weight:700; color:#666; letter-spacing:1px;'>"
            f"RUN PROGRESS (low → close)</div>"
            # Big number + verdict pill on the same line
            f"<div style='display:flex; align-items:baseline; gap:12px; margin-top:6px;'>"
            f"<span style='font-size:30px; font-weight:800; color:{'#27ae60' if run_pct>=0 else '#c0392b'};'>"
            f"{'+' if run_pct>=0 else ''}{run_pct:.2f}%</span>"
            f"<span style='padding:3px 10px; background:{verdict_bg}; color:{verdict_fg}; "
            f"font-size:11px; font-weight:700; border-radius:12px; letter-spacing:0.5px;'>"
            f"{verdict_txt}</span></div>"
            # Sub-line explaining what it means
            f"<div style='font-size:11px; color:#888; margin-top:4px;'>"
            f"from run low {run_low_close:,.1f} on {pd.Timestamp(run_low_date).date()}</div>"
            # Hist benchmark — explicitly labelled
            f"<div style='font-size:12px; color:#444; margin-top:10px; "
            f"padding-top:8px; border-top:1px solid #e8eaee;'>"
            f"Hist {run_label} runs (low→close): median <b>{mag_med:+.2f}%</b> · "
            f"P75 <b>{mag_p75:+.2f}%</b></div>"
            # Footnote — market-wide 3M floor (clearly separate, italicized)
            f"<div style='font-size:11px; color:#999; margin-top:6px; font-style:italic;'>"
            f"Market context: index is {pct_above_low:+.2f}% above its 3-month low "
            f"({low_val:,.1f} on {pd.Timestamp(low_date).date()})</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


with tab_summary:
    st.subheader("At-a-glance regime read")
    st.caption(
        "**Q1** Latest regime · **Q2** Day-over-day probability shifts (any rival ≥5% flags a yellow warning) · "
        "**Q3** Duration vs train/test medians · **Q4** Current run magnitude + vs 3M low + historical comparison"
    )
    stats_by_name = {
        'VNINDEX':    compute_summary_stats(vnindex),
        'VNMIDCAP':   compute_summary_stats(midcap),
        'VNSMALLCAP': compute_summary_stats(smallcap),
    }
    cols_s = st.columns(3)
    for col, (n, df) in zip(cols_s, [('VNINDEX', vnindex), ('VNMIDCAP', midcap), ('VNSMALLCAP', smallcap)]):
        summary_card(col, n, df, stats_by_name[n])

    # ── Conclusion synthesis ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Conclusion")
    bullet_parts = []
    for name, df in [('VNINDEX', vnindex), ('VNMIDCAP', midcap), ('VNSMALLCAP', smallcap)]:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        label = last['label']
        color = COLORS[label]
        _, run_lab, run_len, run_pct, _, _ = current_run_info(df)
        s = stats_by_name[name]
        hist_runs = s['train_runs'].get(run_lab, []) + s['test_runs'].get(run_lab, [])
        hist_mags = s['train_mags'].get(run_lab, []) + s['test_mags'].get(run_lab, [])
        med_len = int(np.median(hist_runs)) if hist_runs else 0
        med_mag = float(np.median(hist_mags)) if hist_mags else 0.0

        fwd = forward_stats_for_regime(df, label)
        test_t5 = fwd.get(('Test (2025+)', 'T+5'), {})
        train_t5 = fwd.get(('Train (≤2024)', 'T+5'), {})

        # Shift warning
        rivals = []
        for lab, p, p_prev in [('Bull', float(last['p_Bull']), float(prev['p_Bull'])),
                               ('Neutral', float(last['p_Neutral']), float(prev['p_Neutral'])),
                               ('Bear', float(last['p_Bear']), float(prev['p_Bear']))]:
            if lab != label and (p >= 0.05 or (p - p_prev) >= 0.05):
                rivals.append(f"P({lab})={p*100:.0f}%")
        shift_note = f" ⚠ rival rising: {' '.join(rivals)}" if rivals else " ✓ stable"

        # Bias verdict from T+5 test win rate + median return
        if test_t5:
            w, m = test_t5['win'], test_t5['med']
            if w >= 0.55 and m > 0.002:
                verdict = "**Bias: long-favored**"
            elif w <= 0.45 and m < -0.002:
                verdict = "**Bias: avoid / short-favored**"
            else:
                verdict = "**Bias: neutral / no edge**"
        else:
            verdict = "**Bias: insufficient data**"

        bullet_parts.append(
            f"<li style='margin-bottom:14px;'>"
            f"<span style='display:inline-block; padding:2px 10px; background:{color}; "
            f"color:#fff; font-weight:700; border-radius:4px; font-size:11px; letter-spacing:1px;'>"
            f"{name} · {label.upper()}</span>"
            f"<span style='color:#666; font-size:12px; margin-left:8px;'>{shift_note}</span>"
            f"<div style='margin-top:6px; font-size:13px; color:#222; line-height:1.6;'>"
            f"In <b>{label}</b> for <b>{run_len} bars</b> (hist median {med_len}); "
            f"price <b>{run_pct:+.2f}%</b> from run low (hist {label} median {med_mag:+.2f}%). "
            f"Historical T+5 in {label} regime: "
            f"train win <b>{train_t5.get('win',0)*100:.0f}%</b> / median <b>{train_t5.get('med',0)*100:+.2f}%</b>, "
            f"test win <b>{test_t5.get('win',0)*100:.0f}%</b> / median <b>{test_t5.get('med',0)*100:+.2f}%</b> "
            f"(up <b>+{test_t5.get('up',0)*100:.1f}%</b> / dn <b>{test_t5.get('dn',0)*100:.1f}%</b>). "
            f"{verdict}.</div></li>"
        )

    st.markdown(
        "<ul style='list-style:none; padding-left:0;'>" + ''.join(bullet_parts) + "</ul>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Bias rule (heuristic): **long-favored** if test T+5 win rate ≥ 55% and median return > +0.2%. "
        "**Avoid** if win rate ≤ 45% and median < −0.2%. Else **neutral**. "
        "T+5 = open-to-open 5-bar holding. Not financial advice."
    )

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

    # Distribution — per (index, regime): share, mean/std of daily returns, run length
    st.subheader("Regime distribution & statistics (full history)")
    st.caption("Per (index, regime): bar count and share, mean & std of daily Close returns, "
               "median & mean run length in bars.")
    dist_rows = []
    for name, df in [('VNINDEX', vnindex), ('VNMIDCAP', midcap), ('VNSMALLCAP', smallcap)]:
        d = df.sort_values('Date').reset_index(drop=True).copy()
        d['ret'] = d['Close'].pct_change()
        total = len(d)
        # run lengths grouped by label across the full series
        segs = regime_segments(d)
        runs_by_lab = {r: [] for r in REGIMES}
        for s, e, lab in segs:
            length = int(((d['Date'] >= s) & (d['Date'] <= e)).sum())
            runs_by_lab[lab].append(length)
        for regime in REGIMES:
            sub = d[d['label'] == regime]
            bars = len(sub)
            share = bars / total * 100.0 if total else 0.0
            r = sub['ret'].dropna()
            mean_ret = float(r.mean()) * 100.0 if len(r) else 0.0
            std_ret  = float(r.std())  * 100.0 if len(r) else 0.0
            runs = runs_by_lab[regime]
            med_run  = int(np.median(runs)) if runs else 0
            mean_run = float(np.mean(runs)) if runs else 0.0
            dist_rows.append({
                'Index': name,
                'Regime': regime,
                'Bars': bars,
                'Share': share,
                'Mean daily ret': mean_ret,
                'Std daily ret':  std_ret,
                'Median run (bars)': med_run,
                'Mean run (bars)':   round(mean_run, 1),
            })
    ddf = pd.DataFrame(dist_rows)
    st.dataframe(
        ddf,
        hide_index=True,
        use_container_width=True,
        key="dist_table",
        column_config={
            'Bars':              st.column_config.NumberColumn(format='%d'),
            'Share':             st.column_config.ProgressColumn(format='%.1f%%', min_value=0.0, max_value=100.0),
            'Mean daily ret':    st.column_config.NumberColumn(format='%+.3f%%'),
            'Std daily ret':     st.column_config.NumberColumn(format='%.3f%%'),
            'Median run (bars)': st.column_config.NumberColumn(format='%d'),
            'Mean run (bars)':   st.column_config.NumberColumn(format='%.1f'),
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

        # Features unique to each method's top-10 (vs the universal top-8)
        unique_per_method = {
            label: [f for f in top10[m] if f not in u8]
            for m, _, label, _ in metrics_def
        }

        # ── Conclusion at the top ────────────────────────────────────────────
        st.subheader("Conclusion — what's important & where methods disagree")
        st.markdown(
            "<div style='padding:16px 20px; background:#f0f7fb; border-left:5px solid #3498db; "
            "border-radius:8px; font-size:14px; line-height:1.7; color:#222;'>"
            f"<b>Strong consensus:</b> {len(u8)} features appear in <b>every</b> method's top-10 — "
            f"a high-agreement core: {', '.join(f'<code>{f}</code>' for f in u8)}. "
            f"<code>ULT_RSI</code> is rank&nbsp;1 in all 4 methods. The trend family "
            f"(<code>DMI_plusDI/minusDI/ADX</code>) plus momentum (<code>AMACD/AMACD_signal</code>) "
            f"plus a single breadth feature (<code>pct_below_ema200</code>) is what every method picks first.<br><br>"
            f"<b>Where methods disagree (top-10 slot 9–10):</b><br>"
            + "".join(
                f"&nbsp;&nbsp;• <b>{label}</b> adds {', '.join(f'<code>{f}</code>' for f in unique_per_method[label])}<br>"
                for label in ['η²', 'RF Gini', 'SHAP', 'Wass']
                if unique_per_method.get(label)
            )
            + "<br><b>Notable divergence:</b> <b>SHAP</b> is the only method that ranks "
            f"<code>fgn_share_20d</code> in its top-10 — XGBoost finds non-linear value in "
            "foreign-participation that univariate scores (η², Wass) miss. "
            "η²/RF/Wass fill their last slots with <code>BBWP</code> (volatility-expansion) features instead."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

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

        # ── 4 horizontal bar charts in a 2×2 grid (wider per chart → readable values) ──
        st.markdown("---")
        st.subheader("Top-10 by each method (interactive)")
        st.caption(
            "Gold = in all 4 methods' top-10 · Gray = in 3 · Tan = in 2 · Red = unique to one method"
        )

        def _render_chart(col, m, mkey, label, mcol):
            with col:
                top_df = imp.sort_values(m, ascending=False).head(10).iloc[::-1]
                bar_colors = [
                    '#f1c40f' if feat_agree.get(f, 0) == 4 else
                    '#95a5a6' if feat_agree.get(f, 0) == 3 else
                    '#d6c4a3' if feat_agree.get(f, 0) == 2 else
                    '#e74c3c'
                    for f in top_df['feature']
                ]
                vmax = float(top_df[m].max())
                # Choose decimals: small values (RF Gini ≤ 0.2) need 3, others 2 reads clean
                decimals = 3 if vmax < 0.5 else 2
                fig = go.Figure(go.Bar(
                    x=top_df[m],
                    y=top_df['feature'],
                    orientation='h',
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=[f"{v:.{decimals}f}" for v in top_df[m]],
                    textposition='outside',
                    textfont=dict(size=15, family='DM Mono', color='#222'),
                    cliponaxis=False,
                    constraintext='none',
                    hovertemplate='<b>%{y}</b><br>' + label + ' = %{x:.4f}<extra></extra>',
                ))
                fig.update_layout(
                    title=dict(
                        text=f"<b style='color:{mcol}'>{label}</b>",
                        x=0.5,
                        font=dict(size=22, family='DM Sans'),
                    ),
                    height=460,
                    margin=dict(l=10, r=110, t=64, b=28),
                    plot_bgcolor='#fafbfc',
                    xaxis=dict(
                        showgrid=True, gridcolor='#eee',
                        tickfont=dict(size=12, family='DM Sans'),
                        range=[0, vmax * 1.30],  # 30% headroom so outside text labels fit
                    ),
                    yaxis=dict(tickfont=dict(size=14, family='DM Sans')),
                    font=dict(family='DM Sans'),
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'displayModeBar': False},
                    key=f"imp_bar_{mkey}",
                )

        # 2×2 grid
        row1 = st.columns(2)
        row2 = st.columns(2)
        layout_cells = [row1[0], row1[1], row2[0], row2[1]]
        for cell, (m, mkey, label, mcol) in zip(layout_cells, metrics_def):
            _render_chart(cell, m, mkey, label, mcol)

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

# ─────────────────────────────── Tab 4: LambdaMART ──────────────────────────

LM_DIR = os.path.join(ROOT, 'lambdamart')


def _fmt_pct(x, decimals=1):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "—"
    return f"{x * 100:+.{decimals}f}%"


def _fmt_sharpe(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "—"
    return f"{x:.3f}"


def _fmt_mdd(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "—"
    return f"{x * 100:.1f}%"


@st.cache_data
def _load_lm_csv(filename, _mtime_key):
    path = os.path.join(LM_DIR, filename)
    return pd.read_csv(path) if os.path.exists(path) else None


_lm_mtime = tuple(int(os.path.getmtime(os.path.join(LM_DIR, f)))
                  for f in os.listdir(LM_DIR) if os.path.exists(os.path.join(LM_DIR, f))) if os.path.exists(LM_DIR) else ()

with tab_lm:
    st.title("Stock Ranking — LambdaMART Research")
    st.markdown(
        "<div style='display:flex; gap:10px; margin-top:-6px; margin-bottom:8px; align-items:center;'>"
        "<span style='color:#666; font-size:13px;'>Cross-sectional LambdaMART ranking · "
        "XGBoost <code>rank:ndcg</code> · Walk-forward · <code>shares_cash</code> engine</span>"
        "<span style='display:inline-block; padding:3px 10px; background:#27ae60; color:#fff; "
        "font-size:11px; font-weight:700; border-radius:12px; letter-spacing:0.5px;'>"
        "DATA THROUGH 2026-05-12</span></div>",
        unsafe_allow_html=True,
    )

    # ── 1. Header / Status cards ─────────────────────────────────────────
    rob_7f = _load_lm_csv('lambdamart_tr_price7f_robustness_shares_cash_summary.csv', _lm_mtime)
    rob_26 = _load_lm_csv('lambdamart_26f_robustness_shares_cash_summary.csv',       _lm_mtime)
    ms_7f  = _load_lm_csv('lambdamart_tr_price7f_multiseed_full_shares_cash_summary.csv', _lm_mtime)
    ms_26  = _load_lm_csv('lambdamart_26f_multiseed_full_shares_cash_summary.csv',       _lm_mtime)
    yr_7f  = _load_lm_csv('lambdamart_tr_price7f_multiseed_full_shares_cash_yearly.csv', _lm_mtime)
    yr_26  = _load_lm_csv('lambdamart_26f_multiseed_full_shares_cash_yearly.csv',       _lm_mtime)
    holdings_legacy = _load_lm_csv('lambdamart_26f_frozen_latest_portfolio.csv',         _lm_mtime)

    # Headline metrics: 26f EqWt + tr_price7f EqWt (Q75 hidden per user request)
    if rob_26 is not None:
        headline = rob_26[(rob_26['scenario'] == 'TOPK_5') & (rob_26['config'] == 'EqWt')]
        h_26 = headline.iloc[0] if len(headline) else None
    else:
        h_26 = None
    if rob_7f is not None:
        h_7f_row = rob_7f[(rob_7f['scenario'] == 'TOPK_5') & (rob_7f['config'] == 'EqWt')]
        h_7f = h_7f_row.iloc[0] if len(h_7f_row) else None
    else:
        h_7f = None

    c1, c2, c3, c4 = st.columns(4)
    if h_26 is not None:
        c1.metric("26f · Cum return",        _fmt_pct(h_26['cum']),    help="UNIV_FULL · TOPK_5 · RB_5 · shares_cash · EqWt")
        c2.metric("26f · Sharpe",            _fmt_sharpe(h_26['sharpe']))
        c3.metric("26f · Max drawdown",      _fmt_mdd(h_26['mdd']))
    if h_7f is not None:
        c4.metric("tr_price7f · Cum return", _fmt_pct(h_7f['cum']),    help="Same setup, EqWt")

    # Decision summary
    st.markdown(
        "<div style='padding:14px 18px; margin-top:10px; background:#f0f7fb; "
        "border-left:5px solid #3498db; border-radius:8px; font-size:13px; line-height:1.7;'>"
        "<b>Research candidate:</b> cross-sectional LambdaMART ranking on Vietnam equities, "
        "evaluated under <code>shares_cash</code> walk-forward backtest. Two feature schemas compared: "
        "<code>26f</code> (technical baseline) and <code>tr_price7f</code> (simpler price-decomposition). "
        "<span style='display:inline-block; margin-left:6px; padding:2px 8px; background:#f39c12; "
        "color:#fff; font-size:11px; font-weight:700; border-radius:10px;'>RESEARCH ONLY</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.warning("**2026 YTD alpha is negative** for both schemas. This remains a research candidate, "
               "not a production trading signal. Live paper trading, slippage modelling, and capacity "
               "checks are required before any deployment.")

    # ── 2. Latest Holdings (both 7f and 26f side-by-side) ────────────────
    st.markdown("---")
    st.subheader("Latest Holdings — tr_price7f & 26f")

    holdings_7f  = _load_lm_csv('lambdamart_tr_price7f_latest_portfolio.csv', _lm_mtime)
    holdings_26f = _load_lm_csv('lambdamart_26f_latest_portfolio.csv',         _lm_mtime)

    # ONE consolidated status line at the top of this section
    missing = []
    if holdings_7f  is None: missing.append('tr_price7f')
    if holdings_26f is None: missing.append('26f')
    if missing:
        st.info(
            f"Canonical {' & '.join(missing)} holdings CSV"
            f"{'s have' if len(missing)>1 else ' has'} not been regenerated for 2026-05-12. "
            f"{'Legacy 26f frozen monitor (2026-04-17) shown as fallback.' if '26f' in missing else ''} "
            "Holdings should be re-run via the LambdaMART monitor script before presenting live names."
        )

    def _render_holdings(col, df, schema_name, key_suffix, is_legacy=False):
        with col:
            label_color = '#e67e22' if schema_name == 'tr_price7f' else '#3498db'
            badge = (
                f"<span style='display:inline-block; padding:2px 10px; background:{label_color}; "
                f"color:#fff; font-size:11px; font-weight:700; border-radius:10px; letter-spacing:1px;'>"
                f"{schema_name.upper()}</span>"
            )
            if df is None:
                st.markdown(
                    badge + "<span style='margin-left:8px; color:#888; font-size:13px; "
                    "font-style:italic;'>holdings not regenerated yet</span>",
                    unsafe_allow_html=True,
                )
                return
            portfolio_date = df['portfolio_date'].iloc[0] if 'portfolio_date' in df.columns else df['Date'].iloc[0]
            stale_warn = ""
            try:
                pd_date = pd.Timestamp(portfolio_date).date()
                if pd_date < pd.Timestamp('2026-05-12').date():
                    days_stale = (pd.Timestamp('2026-05-12').date() - pd_date).days
                    stale_warn = (
                        f"<span style='display:inline-block; margin-left:8px; padding:2px 8px; "
                        f"background:#f39c12; color:#fff; font-size:10px; font-weight:700; border-radius:8px;'>"
                        f"STALE {days_stale}d</span>"
                    )
            except Exception:
                pass
            legacy_tag = (
                "<span style='display:inline-block; margin-left:8px; padding:2px 8px; "
                "background:#95a5a6; color:#fff; font-size:10px; font-weight:700; border-radius:8px;'>"
                "LEGACY</span>"
            ) if is_legacy else ""
            st.markdown(
                f"{badge}{stale_warn}{legacy_tag}"
                f"<span style='margin-left:10px; color:#666; font-size:12px;'>"
                f"as of <code>{portfolio_date}</code></span>",
                unsafe_allow_html=True,
            )
            disp = df.copy()
            if 'eq_weight' in disp.columns:
                disp['eq_weight']  = disp['eq_weight']  * 100.0
            cols_show = [c for c in ['Ticker','score','eq_weight'] if c in disp.columns]
            st.dataframe(
                disp[cols_show], hide_index=True, use_container_width=True,
                key=f"lm_holdings_{key_suffix}",
                column_config={
                    'score':      st.column_config.NumberColumn('Score',  format='%.5f'),
                    'eq_weight':  st.column_config.ProgressColumn('Weight (EqWt)', format='%.1f%%', min_value=0.0, max_value=100.0),
                },
            )

    # Side-by-side: tr_price7f (primary) | 26f (control)
    hcol_7f, hcol_26 = st.columns(2)
    _render_holdings(hcol_7f, holdings_7f, 'tr_price7f', '7f', is_legacy=False)
    # 26f: prefer canonical; fall back to legacy frozen monitor (clearly tagged)
    if holdings_26f is not None:
        _render_holdings(hcol_26, holdings_26f, '26f', '26f', is_legacy=False)
    else:
        _render_holdings(hcol_26, holdings_legacy, '26f', '26f_legacy', is_legacy=True)

    # ── 3. Backtest Performance (UNIV_FULL / TOPK_5 / RB_5, EqWt) ────────
    st.markdown("---")
    st.subheader("Backtest Performance — default setup")
    st.caption("UNIV_FULL · TOPK_5 · RB_5 · `shares_cash` engine · EqWt · alpha vs VNINDEX")

    if rob_7f is not None and rob_26 is not None:
        def _row(df, scenario, schema):
            sub = df[(df['scenario'] == scenario) & (df['config'] == 'EqWt')]
            if len(sub) == 0:
                return None
            r = sub.iloc[0]
            return {
                'Schema': schema,
                'Cum return':  r['cum'] * 100.0,
                'Sharpe':      float(r['sharpe']),
                'Max DD':      r['mdd'] * 100.0,
                'Alpha total': r['alpha_total'] * 100.0,
                'Beat years':  int(r['beat_years']),
            }
        perf_rows = []
        for schema_df, schema_name in [(rob_26, '26f'), (rob_7f, 'tr_price7f')]:
            row = _row(schema_df, 'TOPK_5', schema_name)
            if row:
                perf_rows.append(row)
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            st.dataframe(
                perf_df, hide_index=True, use_container_width=True, key="lm_perf_main",
                column_config={
                    'Cum return':  st.column_config.NumberColumn(format='%+.1f%%'),
                    'Sharpe':      st.column_config.NumberColumn(format='%.3f'),
                    'Max DD':      st.column_config.NumberColumn(format='%.1f%%'),
                    'Alpha total': st.column_config.NumberColumn(format='%+.1f%%'),
                    'Beat years':  st.column_config.NumberColumn(format='%d'),
                },
            )
            st.caption(
                "**Read:** `26f` slightly beats `tr_price7f` on cumulative return (+194.1% vs +159.0%), "
                "and they have similar max drawdown (-51.6% vs -48.3%) and Sharpe (0.720 vs 0.688). "
                "Both schemas comfortably beat VNINDEX over the test window (+132% to +167% alpha)."
            )

    # ── 4. Yearly Performance (EqWt only) ────────────────────────────────
    st.markdown("---")
    st.subheader("Yearly Performance (5-seed multiseed mean, EqWt)")

    if yr_7f is not None and yr_26 is not None:
        y7  = yr_7f[yr_7f['config']   == 'EqWt'][['year','strategy_return_mean','alpha_mean']]
        y26 = yr_26[yr_26['config']   == 'EqWt'][['year','strategy_return_mean','alpha_mean']]
        y7  = y7.rename(columns={'strategy_return_mean':'tr_price7f Return', 'alpha_mean':'tr_price7f Alpha'})
        y26 = y26.rename(columns={'strategy_return_mean':'26f Return',       'alpha_mean':'26f Alpha'})
        ydf = y26.merge(y7, on='year').sort_values('year').reset_index(drop=True)

        col_r, col_a = st.columns(2)
        with col_r:
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(name='26f',         x=ydf['year'].astype(str), y=ydf['26f Return']         * 100, marker_color='#3498db'))
            fig_r.add_trace(go.Bar(name='tr_price7f',  x=ydf['year'].astype(str), y=ydf['tr_price7f Return']  * 100, marker_color='#e67e22'))
            fig_r.update_layout(
                title="<b>Strategy Return</b>",
                barmode='group', height=340, yaxis_title='% return',
                margin=dict(l=10, r=10, t=44, b=24), plot_bgcolor='#fafbfc',
                font=dict(family='DM Sans'),
            )
            st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar': False}, key="lm_year_ret")
        with col_a:
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(name='26f',         x=ydf['year'].astype(str), y=ydf['26f Alpha']         * 100, marker_color='#3498db'))
            fig_a.add_trace(go.Bar(name='tr_price7f',  x=ydf['year'].astype(str), y=ydf['tr_price7f Alpha']  * 100, marker_color='#e67e22'))
            fig_a.update_layout(
                title="<b>Alpha vs VNINDEX</b>",
                barmode='group', height=340, yaxis_title='% alpha',
                margin=dict(l=10, r=10, t=44, b=24), plot_bgcolor='#fafbfc',
                font=dict(family='DM Sans'),
            )
            fig_a.add_hline(y=0, line=dict(color='#888', width=1, dash='dot'))
            st.plotly_chart(fig_a, use_container_width=True, config={'displayModeBar': False}, key="lm_year_alpha")

        ydisp = ydf.copy()
        for col in ['26f Return','26f Alpha','tr_price7f Return','tr_price7f Alpha']:
            ydisp[col] = ydisp[col] * 100.0
        st.dataframe(
            ydisp, hide_index=True, use_container_width=True, key="lm_year_table",
            column_config={
                'year':              st.column_config.NumberColumn('Year', format='%d'),
                '26f Return':        st.column_config.NumberColumn(format='%+.1f%%'),
                '26f Alpha':         st.column_config.NumberColumn(format='%+.1f%%'),
                'tr_price7f Return': st.column_config.NumberColumn(format='%+.1f%%'),
                'tr_price7f Alpha':  st.column_config.NumberColumn(format='%+.1f%%'),
            },
        )
        st.caption("**2026 YTD is partial year** — interpret accordingly.")

    # ── 5. Robustness Test (EqWt) ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Robustness Test — EqWt by scenario")

    if rob_7f is not None and rob_26 is not None:
        scenarios = ['TOPK_3','TOPK_5','TOPK_10','RB_10','UNIV_TOP90','UNIV_TOP70']
        rob_rows = []
        for sc in scenarios:
            r7 = rob_7f[(rob_7f['scenario'] == sc) & (rob_7f['config'] == 'EqWt')]
            r26 = rob_26[(rob_26['scenario'] == sc) & (rob_26['config'] == 'EqWt')]
            if len(r7) == 0 or len(r26) == 0:
                continue
            r7, r26 = r7.iloc[0], r26.iloc[0]
            rob_rows.append({
                'Scenario':       sc,
                'tr_price7f Cum':    r7['cum']  * 100.0,
                'tr_price7f Sharpe': float(r7['sharpe']),
                'tr_price7f MDD':    r7['mdd']  * 100.0,
                '26f Cum':           r26['cum'] * 100.0,
                '26f Sharpe':        float(r26['sharpe']),
                '26f MDD':           r26['mdd'] * 100.0,
            })
        if rob_rows:
            rdf = pd.DataFrame(rob_rows)
            st.dataframe(
                rdf, hide_index=True, use_container_width=True, key="lm_robust",
                column_config={
                    'tr_price7f Cum':    st.column_config.NumberColumn(format='%+.1f%%'),
                    'tr_price7f Sharpe': st.column_config.NumberColumn(format='%.3f'),
                    'tr_price7f MDD':    st.column_config.NumberColumn(format='%.1f%%'),
                    '26f Cum':           st.column_config.NumberColumn(format='%+.1f%%'),
                    '26f Sharpe':        st.column_config.NumberColumn(format='%.3f'),
                    '26f MDD':           st.column_config.NumberColumn(format='%.1f%%'),
                },
            )
            st.caption(
                "**Robustness read:** results vary materially across scenarios. Both schemas remain "
                "positive in concentrated faster setups (`TOPK_3/5`) and degrade in wider/slower variants. "
                "Universe-reduction (`UNIV_TOP90/70`) stresses both — interpret these as floor conditions."
            )

    # ── 6. Multiseed Sensitivity (EqWt) ─────────────────────────────────
    st.markdown("---")
    st.subheader("Multiseed Sensitivity (5 seeds, EqWt)")

    if ms_7f is not None and ms_26 is not None:
        ms_rows = []
        for schema_df, schema_name in [(ms_26, '26f'), (ms_7f, 'tr_price7f')]:
            sub = schema_df[schema_df['config'] == 'EqWt']
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            ms_rows.append({
                'Schema':           schema_name,
                'Cum mean':         r['cum_mean']         * 100.0,
                'Cum std':          r['cum_std']          * 100.0,
                'Sharpe mean':      float(r['sharpe_mean']),
                'Sharpe std':       float(r['sharpe_std']),
                'MDD mean':         r['mdd_mean']         * 100.0,
                'Beat years mean':  float(r['beat_years_mean']),
            })
        if ms_rows:
            ms_df = pd.DataFrame(ms_rows)
            st.dataframe(
                ms_df, hide_index=True, use_container_width=True, key="lm_multiseed",
                column_config={
                    'Cum mean':        st.column_config.NumberColumn(format='%+.1f%%'),
                    'Cum std':         st.column_config.NumberColumn(format='%.1f%%'),
                    'Sharpe mean':     st.column_config.NumberColumn(format='%.3f'),
                    'Sharpe std':      st.column_config.NumberColumn(format='%.3f'),
                    'MDD mean':        st.column_config.NumberColumn(format='%.1f%%'),
                    'Beat years mean': st.column_config.NumberColumn(format='%.1f'),
                },
            )
            st.caption(
                "**Multiseed read:** `tr_price7f` (preferred) is markedly more stable across random seeds "
                "(Cum std 14.4% vs 52.9% for the `26f` control). The `26f` baseline posts a higher mean "
                "cumulative return (+161.3%) but the wide seed dispersion makes that headline less reliable "
                "in live trading — which is why we lead with `tr_price7f`."
            )

    # ── 7. Big Picture / Methodology ────────────────────────────────────
    st.markdown("---")
    st.subheader("Methodology")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**Model**  \n"
            "LambdaMART is a **cross-sectional stock-ranking** model. Each trading date is one ranking query. "
            "The model ranks eligible stocks by expected forward return; we select the top 5 and convert that "
            "ranking into a long-only portfolio.\n\n"
            "Trained with **XGBoost `rank:ndcg`** (a LambdaMART/LambdaRank objective). Labels are **top-heavy "
            "forward-return grades**, not simple regression targets."
        )
        st.markdown(
            "**Backtest**  \n"
            "Walk-forward protocol governs training/test over time. The **`shares_cash` engine** governs "
            "execution and accounting — tracking shares, cash, open-price execution, transaction fees, "
            "forced holds, and weight drift. Universe is gated by a **10 bn VND point-in-time liquidity** "
            "filter to keep names tradable at realistic size."
        )
    with col_b:
        st.markdown(
            "**Feature schemas**  \n"
            "- `tr_price7f` — **preferred**: simpler raw-price decomposition (overnight, intraday, range)\n"
            "- `26f` — **baseline / control**: rich technical-analysis stack (DMI, BBWP, ULT_RSI, …)\n\n"
            "Both schemas use **equal-weighted top-5** portfolios. `tr_price7f` is preferred for its "
            "seed stability and universe robustness; `26f` is retained as a higher-headline control."
        )
        st.markdown(
            "**Why this matters**  \n"
            "Cross-sectional ranking models like LambdaMART can extract alpha from daily security relative-ordering. "
            "But success depends on the operating regime (universe size, rebalance speed, portfolio width). "
            "The tables above stress-test the model across those dimensions."
        )

    # ── 8. Final Conclusion box ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='padding:18px 22px; background:#fff8e1; border:2px solid #f39c12; "
        "border-radius:10px; font-size:14px; line-height:1.8;'>"
        "<div style='font-size:13px; font-weight:700; color:#b9770e; letter-spacing:1px; margin-bottom:6px;'>"
        "FINAL CONCLUSION</div>"
        "<b>Final framing:</b> <code>tr_price7f</code> EqWt is the <b>preferred schema</b>; "
        "<code>26f</code> EqWt is the <b>baseline / control</b>. "
        "Both beat VNINDEX over the test window (+132% to +167% alpha, EqWt) under the "
        "10 bn VND point-in-time liquidity gate with <code>shares_cash</code> execution. "
        "<code>tr_price7f</code> is chosen for its seed stability and universe robustness; "
        "<code>26f</code> posts a higher headline but with much wider seed dispersion.<br><br>"
        "<b>Do not treat this as production-ready.</b> 2026 YTD alpha is negative for both schemas. "
        "Drawdowns are still large (-48% to -52% in default setup). Performance weakens at slower rebalancing "
        "and in wider books. The edge is regime-dependent. Live paper trading, stricter transaction-cost "
        "modeling, capacity checks, and repeated audit-harness validation are required before any deployment."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"<div style='text-align:center; color:#bbb; font-size:11px; margin-top:30px; "
    f"font-family: DM Mono, monospace;'>"
    f"CJM v7 (26 features) · K=3 · λ=50 · grid=0.05 · "
    f"Data through {max(df['Date'].max() for df in indices.values()).date()}</div>",
    unsafe_allow_html=True,
)
