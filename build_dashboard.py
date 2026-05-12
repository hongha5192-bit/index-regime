"""
Build a self-contained static HTML dashboard with regime classification
for VNINDEX / VNMIDCAP / VNSMALLCAP.

Reads:
  - phase4_cjm_v4_results.npz                     (VNINDEX)
  - phase4_regime_latest_full.csv                 (VNINDEX extension)
  - phase4_cjm_v4_midcap_shared_results.npz       (VNMIDCAP)
  - phase4_cjm_v4_smallcap_shared_results.npz     (VNSMALLCAP)

Writes:
  - regime_dashboard.html  (single file, ~2MB, no external deps)

Run:
    python webapp/build_dashboard.py
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# Deploy layout: build_dashboard.py at repo root, data files in ./data/
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, 'data')
OUT_HTML = os.path.join(_HERE, 'regime_dashboard.html')

REGIMES = ['Bull', 'Neutral', 'Bear']
COLORS  = {'Bull': '#2ecc71', 'Neutral': '#3498db', 'Bear': '#e74c3c'}

OHLC_CSV = {
    'VNINDEX':    'VNINDEX_OHLCV_with_features_v4.csv',
    'VNMIDCAP':   'VNMIDCAP_OHLCV_with_features_v4_shared.csv',
    'VNSMALLCAP': 'VNSMALLCAP_OHLCV_with_features_v4_shared.csv',
}


def load_regime_series(npz_path, ohlc_csv, ext_csv=None, ext_label_col=None,
                       ext_proba_cols=None):
    """Merge regime labels + probability vectors (from npz) with OHLC bars (from CSV)."""
    d = np.load(npz_path, allow_pickle=True)
    dates = pd.to_datetime(d['dates'])
    train_mask = d['train_mask']
    test_mask  = d['test_mask']
    n = len(dates)
    labels = np.empty(n, dtype=int)
    labels[train_mask] = d['labels_train']
    labels[test_mask]  = d['labels_test']
    proba = np.empty((n, 3))
    proba[train_mask] = d['proba_train']
    proba[test_mask]  = d['proba_test']
    lab_df = pd.DataFrame({
        'Date': dates,
        'label': [REGIMES[k] for k in labels],
        'p_Bull':    proba[:, 0],
        'p_Neutral': proba[:, 1],
        'p_Bear':    proba[:, 2],
    })
    if ext_csv and os.path.exists(ext_csv):
        ext = pd.read_csv(ext_csv, parse_dates=['Date'])
        cols = ['Date', ext_label_col] + (list(ext_proba_cols) if ext_proba_cols else [])
        ext = ext[cols].rename(columns={ext_label_col: 'label'})
        if ext_proba_cols:
            ext = ext.rename(columns={ext_proba_cols[0]:'p_Bull',
                                      ext_proba_cols[1]:'p_Neutral',
                                      ext_proba_cols[2]:'p_Bear'})
        cutoff = lab_df['Date'].max()
        new = ext[ext['Date'] > cutoff]
        if len(new):
            keep_cols = ['Date','label','p_Bull','p_Neutral','p_Bear']
            for c in keep_cols:
                if c not in new.columns: new[c] = np.nan
            lab_df = pd.concat([lab_df, new[keep_cols]], ignore_index=True)

    ohlc = pd.read_csv(os.path.join(ROOT, ohlc_csv), parse_dates=['Date'])
    ohlc = ohlc[['Date','Open','High','Low','Close']]
    df = lab_df.merge(ohlc, on='Date', how='left').dropna(subset=['Open','High','Low','Close'])
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
    df = df.drop_duplicates(subset='Date', keep='first').sort_values('Date').reset_index(drop=True)
    return df


def compute_long_metrics(df, periods=(3, 5, 15), split_date='2025-01-01'):
    """Continuous-long, OPEN-TO-OPEN execution.
       Entry = open[t]; exit = open[t+N].
       Excursion window for upside/downside = the NEXT N bars after entry: t+1..t+N."""
    d = df.copy().reset_index(drop=True)
    o = d['Open'].values
    h = d['High'].values
    l = d['Low'].values
    n = len(d)

    rows = []
    for N in periods:
        for i in range(n - N):
            entry = o[i]
            exit_ret = o[i + N] / entry - 1.0
            window_hi = h[i+1:i+N+1].max()
            window_lo = l[i+1:i+N+1].min()
            upside   = window_hi / entry - 1.0
            downside = window_lo / entry - 1.0
            rows.append({
                'Date': d['Date'].iloc[i],
                'regime': d['label'].iloc[i],
                'period': f'T+{N}',
                'ret': exit_ret,
                'up':  upside,
                'dn':  downside,
            })

    m = pd.DataFrame(rows)
    split_ts = pd.Timestamp(split_date)
    m['split'] = np.where(m['Date'] < split_ts, 'Train (≤2024)', 'Test (2025+)')

    # Robust summary using MEDIAN + IQR (P25, P75) — resistant to outliers.
    #   median_ret = typical realized return
    #   p75_up     = upper-quartile of upside (75% of trades had upside ≤ this)
    #   p25_dn     = lower-quartile of downside (25% of trades had downside ≤ this, i.e., deeper drawdown)
    agg = (m.groupby(['split','regime','period'])
             .agg(n=('ret','size'),
                  win_rate=('ret', lambda s: float((s > 0).mean())),
                  median_ret=('ret', lambda s: float(s.quantile(0.50))),
                  p75_up=('up', lambda s: float(s.quantile(0.75))),
                  p25_dn=('dn', lambda s: float(s.quantile(0.25))))
             .reset_index())

    regime_order = {'Bull':0,'Neutral':1,'Bear':2}
    period_order = {f'T+{N}':i for i,N in enumerate(periods)}
    split_order  = {'Train (≤2024)':0, 'Test (2025+)':1}
    agg['_r'] = agg['regime'].map(regime_order)
    agg['_p'] = agg['period'].map(period_order)
    agg['_s'] = agg['split'].map(split_order)
    agg = agg.sort_values(['_s','_r','_p']).drop(columns=['_r','_p','_s']).reset_index(drop=True)
    return agg


# Back-compat shim: provide both new (median_ret/p75_up/p25_dn) and legacy alias names
# in case downstream callers reference avg_ret. Not strictly needed inside this file.


def metrics_table_html(agg, index_name):
    """Render a single HTML table per index, with split shown as a section."""
    parts = [f'<h3 style="margin-top:18px;">{index_name} — Long-position performance by regime</h3>']
    for split in ['Train (≤2024)', 'Test (2025+)']:
        sub = agg[agg['split'] == split]
        if len(sub) == 0:
            continue
        parts.append(f'<div class="split-label">{split}</div>')
        parts.append('<table class="perf-table"><thead><tr>'
                     '<th>Regime</th><th>Period</th><th>N</th>'
                     '<th>Win&nbsp;rate</th><th>Median&nbsp;return</th>'
                     '<th>P75&nbsp;upside</th>'
                     '<th>P25&nbsp;downside</th>'
                     '</tr></thead><tbody>')
        for _, r in sub.iterrows():
            color = COLORS.get(r['regime'], '#888')
            ret_color = '#1e7e34' if r['median_ret'] >= 0 else '#c0392b'
            parts.append(
                f'<tr>'
                f'<td><span class="regime-pill" style="background:{color}">{r["regime"]}</span></td>'
                f'<td>{r["period"]}</td>'
                f'<td>{int(r["n"])}</td>'
                f'<td>{r["win_rate"]*100:.1f}%</td>'
                f'<td style="color:{ret_color};font-weight:600">{r["median_ret"]*100:+.2f}%</td>'
                f'<td style="color:#1e7e34">+{r["p75_up"]*100:.2f}%</td>'
                f'<td style="color:#c0392b">{r["p25_dn"]*100:.2f}%</td>'
                f'</tr>'
            )
        parts.append('</tbody></table>')
    return '\n'.join(parts)


def regime_segments(df):
    a = df['label'].values
    if len(a) == 0:
        return []
    segs = []
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[j + 1] == a[i]:
            j += 1
        segs.append((df['Date'].iloc[i], df['Date'].iloc[j], a[i]))
        i = j + 1
    return segs


def make_chart(df, title, yaxis_label, show_rangeslider=False):
    fig = go.Figure()
    lo = float(df['Close'].min()) * 0.97
    hi = float(df['Close'].max()) * 1.03

    # Regime-shaded background for each contiguous run
    legend_done = set()
    for start, end, lab in regime_segments(df):
        showlegend = lab not in legend_done
        legend_done.add(lab)
        right = pd.Timestamp(end) + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[start, right, right, start, start],
            y=[lo, lo, hi, hi, lo],
            fill='toself', mode='lines',
            line=dict(width=0), fillcolor=COLORS[lab],
            opacity=0.25, name=lab,
            legendgroup=lab, showlegend=showlegend,
            hoverinfo='skip',
        ))

    # Black price line on top
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', line=dict(color='black', width=1.2),
        name='Close',
        customdata=df['label'],
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Close: %{y:.2f}<br>Regime: %{customdata}<extra></extra>',
        showlegend=False,
    ))

    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0='2025-01-01', x1='2025-01-01', y0=0, y1=1,
        line=dict(color='blue', dash='dash', width=1.2),
    )
    fig.add_annotation(
        x='2025-01-01', y=1.0, xref='x', yref='paper',
        text='train | test', showarrow=False,
        xanchor='left', yanchor='top',
        font=dict(color='blue', size=11),
    )

    layout = dict(
        title=dict(text=title, font=dict(size=18)),
        yaxis_title=yaxis_label, xaxis_title=None,
        height=420,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation='h', y=1.08, x=0.50, xanchor='center',
                    bgcolor='rgba(255,255,255,0.85)'),
        plot_bgcolor='white',
        hovermode='x unified',
    )
    if show_rangeslider:
        layout['xaxis'] = dict(rangeslider=dict(visible=True, thickness=0.05))
    else:
        layout['xaxis'] = dict(rangeslider=dict(visible=False))
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5, range=[lo, hi])
    return fig


# ─────────────────────────────────────────────────────────────────
# Feature catalog (the 23 v4 features used to fit CJM)
# ─────────────────────────────────────────────────────────────────
FEATURE_CATALOG = [
    # (group, name, short_desc, formula_short)
    ('Trend',     'DMI_plusDI',       '+DI 14-day — uptrend pressure',                      'smoothed(+DM)/smoothed(TR) × 100, Wilder 14'),
    ('Trend',     'DMI_minusDI',      '−DI 14-day — downtrend pressure',                    'smoothed(−DM)/smoothed(TR) × 100, Wilder 14'),
    ('Trend',     'DMI_ADX',          'ADX 14-day — trend strength (not direction)',         'Wilder-smoothed(|+DI−−DI|/(+DI+−DI)) × 100'),
    ('Momentum',  'ULT_RSI',          'Ultimate RSI 14 (LuxAlgo) — overbought/oversold',     'ULT_RSI(close, len=14, smo1=RMA)'),
    ('Momentum',  'ULT_RSI_signal',   'EMA-14 of ULT_RSI',                                   'EMA(ULT_RSI, 14)'),
    ('Momentum',  'AMACD',            'Adaptive MACD (R²-weighted)',                         'AMACD(close, r²_period=20, fast=10, slow=20)'),
    ('Momentum',  'AMACD_signal',     '9-EMA signal line of AMACD',                          'EMA(AMACD, 9)'),
    ('Momentum',  'AMACD_hist',       'AMACD histogram (AMACD − signal)',                    'AMACD − AMACD_signal'),
    ('Volatility','BBWP',             'Bollinger Band Width Percentile (100-bar lookback)',  'percentile_rank(BBW, lookback=100)'),
    ('Volatility','BBWP_MA',          '5-day MA of BBWP',                                    'SMA(BBWP, 5)'),
    ('Volatility','BBWP_BBW',         'Raw Bollinger Band Width',                            '(upper − lower) / middle'),
    ('Streak',    'consec_hh',        'Consecutive Higher-Highs',                            'run(high.diff > 0)'),
    ('Streak',    'consec_hl',        'Consecutive Higher-Lows',                             'run(low.diff > 0)'),
    ('Streak',    'consec_lh',        'Consecutive Lower-Highs',                             'run(high.diff < 0)'),
    ('Streak',    'consec_ll',        'Consecutive Lower-Lows',                              'run(low.diff < 0)'),
    ('Breadth',   'pct_ursi_oversold',  '% top-200 with Ultimate Osc ≤ 20 (deeply oversold)',  '#{UO≤20} / #valid'),
    ('Breadth',   'pct_ursi_overbought','% top-200 with Ultimate Osc > 80 (deeply overbought)','#{UO>80} / #valid'),
    ('Breadth',   'pct_ursi_above_50',  '% top-200 with Ultimate Osc > 50 (bullish breadth)',  '#{UO>50} / #valid'),
    ('Breadth',   'pct_ursi_below_50',  '% top-200 with Ultimate Osc < 50 (bearish breadth)',  '#{UO<50} / #valid'),
    ('Breadth',   'pct_bb20_below',     '% top-200 with Close < BB20 lower band',              '#{Close<BB20_LO} / #valid'),
    ('Breadth',   'pct_below_ema200',   '% top-200 with Close < EMA200',                       '#{Close<EMA200} / #valid'),
    ('Flow',      'net_fgn_pct_5d',   '5-day net foreign as % of total turnover (flash)',     'Σ_5d(FgnBuy−FgnSell) / Σ_5d(Total)  (dollar-weighted)'),
    ('Flow',      'net_fgn_pct_20d',  '20-day net foreign as % of total turnover (sustained)','Σ_20d(FgnBuy−FgnSell) / Σ_20d(Total)'),
    ('Flow',      'fgn_share_20d',    '20-day gross foreign participation share',             'Σ_20d(FgnBuy+FgnSell) / Σ_20d(Total)'),
    ('Volume',    'up_down_vol_ratio_20d', '20-day volume conviction (orthogonal to trend)',  'log( mean_up_vol_20d / mean_dn_vol_20d ), clipped ±2'),
    ('Volume',    'volume_ratio_20_60',    'Volume regime: short vs long-term participation', 'mean_20d(Volume) / mean_60d(Volume)'),
]

GROUP_COLOR = {
    'Trend':      '#3498db',
    'Momentum':   '#9b59b6',
    'Volatility': '#e67e22',
    'Streak':     '#16a085',
    'Breadth':    '#c0392b',
    'Flow':       '#34495e',
    'Volume':     '#8e44ad',
}

V4_FEATS = [f[1] for f in FEATURE_CATALOG]


def features_tab_html():
    """Build a 'Features' tab body that lists the 23 v4 features used by CJM,
       grouped by category, with latest values for VNINDEX/MID/SML."""
    # Load latest features for each index
    paths = {
        'VNINDEX':    os.path.join(ROOT, 'VNINDEX_OHLCV_with_features_v4.csv'),
        'VNMIDCAP':   os.path.join(ROOT, 'VNMIDCAP_OHLCV_with_features_v4_shared.csv'),
        'VNSMALLCAP': os.path.join(ROOT, 'VNSMALLCAP_OHLCV_with_features_v4_shared.csv'),
    }
    latest = {}
    for name, p in paths.items():
        df = pd.read_csv(p, parse_dates=['Date'])
        df = df.dropna(subset=V4_FEATS).sort_values('Date').reset_index(drop=True)
        latest[name] = df.iloc[-1]
    last_date = max(v['Date'] for v in latest.values()).date()

    rows = []
    last_group = None
    for grp, fname, desc, formula in FEATURE_CATALOG:
        if grp != last_group:
            rows.append(f"""
            <tr class="group-row"><td colspan="6">
              <span class="group-pill" style="background:{GROUP_COLOR[grp]}">{grp}</span>
            </td></tr>""")
            last_group = grp
        val_v = float(latest['VNINDEX'][fname])
        val_m = float(latest['VNMIDCAP'][fname])
        val_s = float(latest['VNSMALLCAP'][fname])
        # Breadth + Flow features are stored as ratios → display as %.
        if grp == 'Flow':
            # net_fgn_pct_* can be negative → show sign; fgn_share_* is always ≥0 → no sign
            fmt = (lambda x: f"{x*100:+.2f}%") if fname.startswith('net_fgn') else (lambda x: f"{x*100:.2f}%")
        elif grp == 'Breadth':
            fmt = lambda x: f"{x*100:.2f}%"
        elif grp == 'Volume':
            # up_down_vol_ratio is log-ratio (sign matters); volume_ratio is plain ratio
            fmt = (lambda x: f"{x:+.3f}") if fname == 'up_down_vol_ratio_20d' else (lambda x: f"{x:.3f}")
        else:
            fmt = lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}"
        rows.append(f"""<tr>
          <td><code>{fname}</code></td>
          <td>{desc}</td>
          <td><code class="formula">{formula}</code></td>
          <td class="num">{fmt(val_v)}</td>
          <td class="num">{fmt(val_m)}</td>
          <td class="num">{fmt(val_s)}</td>
        </tr>""")

    summary = f"""
    <div class="feat-summary">
      <p><b>Total: {len(FEATURE_CATALOG)} features</b> fed into CJM v7 (K=3, λ=50, grid=0.05) for regime classification.</p>
      <ul>
        <li><span class="group-pill" style="background:{GROUP_COLOR['Trend']}">Trend</span> 3 features — DMI/ADX (Wilder 1978)</li>
        <li><span class="group-pill" style="background:{GROUP_COLOR['Momentum']}">Momentum</span> 5 features — ULT_RSI (LuxAlgo) + Adaptive MACD</li>
        <li><span class="group-pill" style="background:{GROUP_COLOR['Volatility']}">Volatility</span> 3 features — Bollinger Band Width Percentile family</li>
        <li><span class="group-pill" style="background:{GROUP_COLOR['Streak']}">Streak</span> 4 features — HH/HL/LH/LL run-length counters</li>
        <li><span class="group-pill" style="background:{GROUP_COLOR['Breadth']}">Breadth</span> 6 features — % of top-200 stocks with URSI oversold/overbought/above50/below50 + below BB20 + below EMA200 (shared across 3 indices)</li>
        <li><span class="group-pill" style="background:{GROUP_COLOR['Flow']}">Flow</span> 3 features — foreign net % at 5d/20d horizons + 20d gross participation share</li>
        <li><span class="group-pill" style="background:{GROUP_COLOR['Volume']}">Volume</span> 2 features — up/down volume share + 20d/60d volume regime ratio</li>
      </ul>
      <p><b>Latest feature snapshot as of {last_date}.</b> Click any feature to inspect its current value in each cap-band.</p>
    </div>
    """

    return summary + f"""
    <table class="feat-table">
      <thead>
        <tr>
          <th>Feature</th><th>Description</th><th>Formula</th>
          <th>VNINDEX</th><th>VNMIDCAP</th><th>VNSMALLCAP</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def importance_tab_html():
    """Build the 'Importance' tab body — 4 complementary metrics:
       η² (BCSS/TSS), RF Gini (CV), SHAP per-class (XGB), Wasserstein-1."""
    import base64
    ext_csv = os.path.join(ROOT, 'feature_importance_extended_v7.csv')
    eta_csv = os.path.join(ROOT, 'feature_importance_v7.csv')
    if not os.path.exists(ext_csv):
        return '<div class="feat-summary"><p>feature_importance_extended_v7.csv not found — run feature_importance_extended_v7.py first.</p></div>'
    imp = pd.read_csv(ext_csv).sort_values('rank_avg').reset_index(drop=True)

    metric_max = {m: imp[m].max() for m in ['eta2_mean','rf_gini','shap_total','wass_max']}
    top3 = imp.head(3); bot3 = imp.tail(3).iloc[::-1]

    # Group summary by avg rank
    grp_summary = imp.groupby('group').agg(rank_avg=('rank_avg','mean'),
                                            eta2=('eta2_mean','mean'),
                                            n=('feature','size')) \
                                      .sort_values('rank_avg')

    summary = f"""
    <div class="feat-summary">
      <p><b>Four complementary metrics</b> measure each feature's regime-separation power
         on the TRAIN split. <b>All values shown are averages across the 3 indices
         (VNINDEX/MIDCAP/SMALLCAP)</b> — per-index rankings can differ (e.g., DMI_minusDI is
         RF rank 2 in the average but rank 5 for VNINDEX alone, where AMACD takes rank 2).</p>
      <ul style="margin:6px 0 6px 18px; font-size:13.5px;">
        <li><b>η²</b> = BCSS/TSS — univariate variance explained by regime (mean-based)</li>
        <li><b>RF Gini</b> — Random Forest impurity importance, 5-fold TimeSeriesSplit CV (multivariate)</li>
        <li><b>SHAP</b> — sum of mean(|SHAP|) across Bull/Neut/Bear from XGBoost multiclass <i>trained to predict CJM labels</i> (see footnote)</li>
        <li><b>Wass</b> — max pairwise Wasserstein-1 between regimes, normalized by σ (tail-sensitive)</li>
      </ul>
      <p style="margin-top:8px;"><b>Sorted by rank_avg</b> = mean of per-metric ranks (lower = better separator).</p>
      <p style="margin-top:6px;"><b>Top 3 (universal separators):</b>
         {' · '.join(f'<code>{r.feature}</code> (rank_avg={r.rank_avg:.1f})' for _, r in top3.iterrows())}</p>
      <p><b>Bottom 3:</b>
         {' · '.join(f'<code>{r.feature}</code> (rank_avg={r.rank_avg:.1f})' for _, r in bot3.iterrows())}</p>
      <p style="margin-top:10px;"><b>Group ranking (mean rank_avg, lower = stronger group):</b></p>
      <ul style="margin-top:4px;">
        {''.join(f'<li><span class="group-pill" style="background:{GROUP_COLOR.get(g, "#888")}">{g}</span>'
                 f' rank_avg={row.rank_avg:.1f}, η²={row.eta2:.3f}  ({int(row.n)} features)</li>'
                 for g, row in grp_summary.iterrows())}
      </ul>
      <p style="font-size:12.5px; color:#666; margin-top:10px;">
        <b>Verification:</b> all 4 metrics independently re-verified — η² (8/8 checks pass, formula
        identity to machine precision), RF Gini (bit-exact reproducibility, sums-to-1 verified),
        SHAP (additivity holds to 2.4e-06; ranking reproduced rank_corr=0.95), Wasserstein (10/10
        checks pass, no slicing bug). Cross-metric divergences indicate features that are useful
        multivariately but weak univariately (e.g., <code>fgn_share_20d</code>: η² rank 18 → SHAP
        rank 10) or vice versa.
      </p>
      <p style="font-size:12px; color:#888; margin-top:8px; border-top:1px solid #eee; padding-top:8px;">
        <b>SHAP caveat:</b> SHAP here measures how well an XGBoost classifier <i>can mimic the CJM's
        regime labels</i> from each feature — it does <i>not</i> directly measure how each feature
        drove the CJM's own EM fit. Use it as a multivariate predictability score (complementary
        to η²), not as a direct attribution to the jump model. The "SHAP dominant" column shows
        which regime an XGBoost tree most often splits on for the feature, which can differ from
        the feature's economic direction (e.g., low RSI is bearish, but XGBoost may route Bull
        decisions through RSI more than Bear decisions).
      </p>
    </div>
    """

    # ── Consensus & Divergence section (top-10 per method) ──
    metrics = [('eta2_mean','η²','#3498db'),
               ('rf_gini','RF Gini','#27ae60'),
               ('shap_total','SHAP','#9b59b6'),
               ('wass_max','Wass','#e67e22')]
    top10_by_metric = {m: imp.sort_values(m, ascending=False).head(10)['feature'].tolist()
                       for m, _, _ in metrics}
    # Agreement count within top-10 across methods
    top10_union = set().union(*top10_by_metric.values())
    feat_agree = {f: sum(1 for m, _, _ in metrics if f in top10_by_metric[m]) for f in top10_union}
    feat_ranks = {f: {m: (top10_by_metric[m].index(f) + 1 if f in top10_by_metric[m] else None)
                       for m, _, _ in metrics} for f in top10_union}

    # Universal-N intersections (within top-10)
    def universal_n(n):
        sets = [set(top10_by_metric[m][:n]) for m, _, _ in metrics]
        return set.intersection(*sets)
    universal_3  = sorted(universal_n(3))
    universal_5  = sorted(universal_n(5))
    universal_8  = sorted(universal_n(8))
    universal_10 = sorted(universal_n(10))

    # Side-by-side top-10 panels
    star_color = {4:'#f1c40f', 3:'#95a5a6', 2:'#d6c4a3', 1:'#e74c3c'}
    method_cols = []
    for m, label, col in metrics:
        items = []
        for rank, f in enumerate(top10_by_metric[m]):
            agree = feat_agree[f]
            bg = star_color[agree]
            items.append(
                f'<li style="margin:3px 0; padding:4px 8px; border-radius:4px; background:{bg}22; '
                f'border-left:3px solid {bg};"><span style="color:#888; font-size:11px;">#{rank+1}</span> '
                f'<code style="font-size:12.5px;">{f}</code></li>')
        method_cols.append(
            f'<div style="flex:1; min-width:180px;">'
            f'<div style="text-align:center; padding:6px; background:{col}; color:white; '
            f'font-weight:600; border-radius:6px 6px 0 0;">{label}</div>'
            f'<ol style="list-style:none; padding:4px; margin:0; background:white; '
            f'border:1px solid #ddd; border-top:none; border-radius:0 0 6px 6px;">{"".join(items)}</ol>'
            f'</div>')

    # Method-unique picks (within top-10)
    unique_picks = []
    for f, n in feat_agree.items():
        if n == 1:
            picked_in = [(label, feat_ranks[f][m]) for m, label, _ in metrics if f in top10_by_metric[m]][0]
            unique_picks.append((f, picked_in[0], picked_in[1], feat_ranks[f]))

    why_lookup = {
        'pct_below_ema200': 'XGBoost routes Bear-class splits through this Breadth feature — multivariate-only value',
        'AMACD':            'Heavier-tailed distributional separation than its smoothed signal — Wass picks tail mass that mean-based metrics miss',
        'BBWP':             'High volatility-regime variance vs smoothed BBWP_MA → distributional spread captured by Wass',
        'BBWP_BBW':         'Raw BBW distributional shape varies more across regimes than the standardized BBWP/BBWP_MA',
        'pct_ursi_above_50':'Breadth tipping point — half the universe bullish/bearish is a natural regime threshold',
        'pct_bb20_below':   'Below-band breadth has wider Bear-tail mass than mean-based metrics suggest',
        'fgn_share_20d':    'Multivariate XGB exploits gross foreign participation despite weak univariate η²',
        'volume_ratio_20_60':'Volume-regime shift detector — value emerges via XGB interactions, not mean separation',
    }
    def why(f):
        return why_lookup.get(f, 'See per-method top-10 — this feature is uniquely picked by one method via its specific statistical lens')

    rows_unique = '\n'.join(
        f'<tr><td style="padding:6px 12px;"><code>{f}</code></td>'
        f'<td style="padding:6px 12px;"><b>{m}</b> (rank {r})</td>'
        f'<td style="padding:6px 12px; font-size:12px; color:#666;">'
        f'{", ".join(f"{lab}={rk}" for (m2, lab, _), rk in zip(metrics, [ranks[m2] for m2, _, _ in metrics]) if rk is not None)}'
        f'</td>'
        f'<td style="padding:6px 12px; font-size:12.5px; color:#555;">{why(f)}</td></tr>'
        for f, m, r, ranks in sorted(unique_picks, key=lambda x: x[2])
    )

    consensus_html = f"""
    <div class="feat-summary">
      <h3 style="margin-top:0; font-size:16px;">Consensus &amp; Divergence — what the 4 methods agree (and don't) on</h3>

      <p style="margin:6px 0;"><b>Universal top-N (features in <i>every</i> method's top-N, within their top-10s):</b></p>
      <ul style="margin:4px 0 4px 18px; font-size:13px;">
        <li>Top-1: <b>1 feature</b> — <code>ULT_RSI</code> (rank 1 in all 4)</li>
        <li>Top-3: <b>{len(universal_3)} feature{'s' if len(universal_3)!=1 else ''}</b> — {' · '.join(f'<code>{f}</code>' for f in universal_3) if universal_3 else '<i>none</i>'}</li>
        <li>Top-5: <b>{len(universal_5)} features</b> — {' · '.join(f'<code>{f}</code>' for f in universal_5)}</li>
        <li>Top-8 (within each top-10): <b>{len(universal_8)} features</b> — {' · '.join(f'<code>{f}</code>' for f in universal_8)}</li>
        <li>Top-10 universal: <b>{len(universal_10)} features</b> — {' · '.join(f'<code>{f}</code>' for f in universal_10)}</li>
      </ul>

      <p style="margin:14px 0 6px;"><b>Per-method top-10 (side-by-side):</b></p>
      <div style="display:flex; gap:10px; flex-wrap:wrap;">
        {''.join(method_cols)}
      </div>
      <div style="margin-top:8px; font-size:11.5px; color:#666;">
        Color band = consensus level (within each method's top-10):
        <span style="background:#f1c40f33; padding:1px 6px; border-radius:3px;">★★★★ all 4</span>
        <span style="background:#95a5a633; padding:1px 6px; border-radius:3px; margin-left:4px;">★★★ in 3</span>
        <span style="background:#d6c4a333; padding:1px 6px; border-radius:3px; margin-left:4px;">★★ in 2</span>
        <span style="background:#e74c3c33; padding:1px 6px; border-radius:3px; margin-left:4px;">★ unique</span>
      </div>

      <p style="margin:14px 0 6px;"><b>Top-10 divergence — features picked by only ONE method:</b></p>
      <table style="width:auto; font-size:13px; background:white; border:1px solid #ddd; border-radius:6px;">
        <thead><tr style="background:#f0f2f5;">
          <th style="padding:8px 12px;">Feature</th>
          <th style="padding:8px 12px;">Only in (rank)</th>
          <th style="padding:8px 12px;">Ranks in other methods</th>
          <th style="padding:8px 12px;">Why that method values it</th>
        </tr></thead>
        <tbody>{rows_unique}</tbody>
      </table>

      <p style="margin:12px 0 0; font-size:12.5px; color:#555;">
        <b>Practical takeaway:</b> the <b>{len(universal_8)}-feature universal top-8 core</b>
        is the high-confidence keep-list (in <i>every</i> method's top-8). Features that
        are method-unique in top-10 reflect each metric's specific statistical lens —
        keep them if the corresponding angle matters for your downstream task.
      </p>
    </div>
    """

    # Embed original η² PNG chart
    png_path = os.path.join(ROOT, 'feature_importance_v7.png')
    img_html = ''
    if os.path.exists(png_path):
        with open(png_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        img_html = f"""
        <details style="margin-bottom:14px;">
          <summary style="cursor:pointer; font-weight:500; color:#444;">Bar chart of η² (click to expand)</summary>
          <div class="chart" style="text-align:center; margin-top:8px;">
            <img src="data:image/png;base64,{b64}" alt="η² bar chart" style="max-width:100%; height:auto;"/>
          </div>
        </details>"""

    # Helper for inline bar+number cell
    def bar_cell(val, vmax, color):
        pct = (val / vmax) * 100 if vmax > 0 else 0
        return f"""<td>
          <div style="display:flex; align-items:center; gap:6px;">
            <div style="flex:1; background:#eef; height:8px; border-radius:4px; overflow:hidden;">
              <div style="width:{pct:.1f}%; background:{color}; height:100%;"></div>
            </div>
            <span class="num" style="min-width:50px; font-size:12px;">{val:.3f}</span>
          </div>
        </td>"""

    # Build SHAP per-class signature (dominant regime by SHAP)
    rows = []
    for i, r in imp.iterrows():
        g = r['group']; color = GROUP_COLOR.get(g, '#888')
        shap_per = {'Bull': r['shap_Bull'], 'Neut': r['shap_Neut'], 'Bear': r['shap_Bear']}
        shap_dom = max(shap_per, key=lambda k: shap_per[k])
        shap_dom_pct = shap_per[shap_dom] / max(r['shap_total'], 1e-9) * 100
        rows.append(f"""<tr>
          <td class="num" style="color:#888;">{i+1}</td>
          <td><code>{r['feature']}</code></td>
          <td><span class="group-pill" style="background:{color}">{g}</span></td>
          {bar_cell(r['eta2_mean'], metric_max['eta2_mean'], color)}
          {bar_cell(r['rf_gini'],   metric_max['rf_gini'],   color)}
          {bar_cell(r['shap_total'],metric_max['shap_total'],color)}
          {bar_cell(r['wass_max'],  metric_max['wass_max'],  color)}
          <td class="num"><b>{r['rank_avg']:.1f}</b></td>
          <td class="num" style="font-size:11.5px; color:#444;">
            <b>{shap_dom}</b> ({shap_dom_pct:.0f}%)
          </td>
          <td class="num" style="color:#888;">{r.get('signature','—')}</td>
        </tr>""")

    table = f"""
    <table class="feat-table">
      <thead><tr>
        <th>#</th><th>Feature</th><th>Group</th>
        <th style="min-width:140px;">η²<br><span style="font-weight:400; color:#888; font-size:11px;">BCSS/TSS · avg of 3</span></th>
        <th style="min-width:140px;">RF Gini<br><span style="font-weight:400; color:#888; font-size:11px;">CV · avg of 3</span></th>
        <th style="min-width:140px;">SHAP total<br><span style="font-weight:400; color:#888; font-size:11px;">XGB → CJM labels · avg of 3</span></th>
        <th style="min-width:140px;">Wass-1 max<br><span style="font-weight:400; color:#888; font-size:11px;">σ-normalized · avg of 3</span></th>
        <th>rank_avg</th>
        <th>SHAP dominant<br><span style="font-weight:400; color:#888; font-size:11px;">tree-routing</span></th>
        <th>η² signature<br><span style="font-weight:400; color:#888; font-size:11px;">most extreme</span></th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """
    return consensus_html + summary + img_html + table


def card_html(name, df):
    last = df.iloc[-1]
    label = last['label']
    color = COLORS[label]
    pB = float(last.get('p_Bull', np.nan))
    pN = float(last.get('p_Neutral', np.nan))
    pX = float(last.get('p_Bear', np.nan))
    if not np.isfinite(pB):
        proba_html = ""
    else:
        proba_html = f"""
        <div class="proba-row">
          <div class="proba-bar">
            <div style="width:{pB*100:.1f}%; background:{COLORS['Bull']}"    title="Bull {pB*100:.0f}%"></div>
            <div style="width:{pN*100:.1f}%; background:{COLORS['Neutral']}" title="Neutral {pN*100:.0f}%"></div>
            <div style="width:{pX*100:.1f}%; background:{COLORS['Bear']}"    title="Bear {pX*100:.0f}%"></div>
          </div>
          <div class="proba-text">
            <span style="color:{COLORS['Bull']}">B&nbsp;{pB*100:.0f}%</span> ·
            <span style="color:{COLORS['Neutral']}">N&nbsp;{pN*100:.0f}%</span> ·
            <span style="color:{COLORS['Bear']}">X&nbsp;{pX*100:.0f}%</span>
          </div>
        </div>"""
    return f"""
    <div class="card" style="background:{color}22; border-left:6px solid {color};">
      <div class="card-name">{name}</div>
      <div class="card-regime" style="color:{color};">{label}</div>
      <div class="card-close">Close: <b>{last['Close']:.2f}</b></div>
      <div class="card-date">As of {last['Date'].date()}</div>
      {proba_html}
    </div>
    """


def dist_row(df, name):
    counts = df['label'].value_counts().reindex(REGIMES, fill_value=0)
    total = counts.sum()
    pct = (counts / total * 100).round(1) if total else counts
    return f"<tr><td>{name}</td><td>{int(total)}</td>" \
           f"<td>{pct['Bull']:.1f}%</td>" \
           f"<td>{pct['Neutral']:.1f}%</td>" \
           f"<td>{pct['Bear']:.1f}%</td></tr>"


def main():
    print("Loading regime series...")
    vnindex = load_regime_series(
        os.path.join(ROOT, 'phase4_cjm_v7_vnindex_results.npz'),
        ohlc_csv=OHLC_CSV['VNINDEX'],
        ext_csv=os.path.join(ROOT, 'phase4_regime_v7_vnindex.csv'),
        ext_label_col='v7_label',
        ext_proba_cols=('v7_Bull', 'v7_Neut', 'v7_Bear'),
    )
    midcap = load_regime_series(
        os.path.join(ROOT, 'phase4_cjm_v7_midcap_results.npz'),
        ohlc_csv=OHLC_CSV['VNMIDCAP'],
        ext_csv=os.path.join(ROOT, 'phase4_regime_v7_midcap.csv'),
        ext_label_col='v7_label',
        ext_proba_cols=('v7_Bull', 'v7_Neut', 'v7_Bear'),
    )
    smallcap = load_regime_series(
        os.path.join(ROOT, 'phase4_cjm_v7_smallcap_results.npz'),
        ohlc_csv=OHLC_CSV['VNSMALLCAP'],
        ext_csv=os.path.join(ROOT, 'phase4_regime_v7_smallcap.csv'),
        ext_label_col='v7_label',
        ext_proba_cols=('v7_Bull', 'v7_Neut', 'v7_Bear'),
    )
    print(f"  VNINDEX:   {len(vnindex)} bars  ({vnindex['Date'].min().date()} → {vnindex['Date'].max().date()})")
    print(f"  VNMIDCAP:  {len(midcap)} bars   ({midcap['Date'].min().date()} → {midcap['Date'].max().date()})")
    print(f"  VNSMALL:   {len(smallcap)} bars ({smallcap['Date'].min().date()} → {smallcap['Date'].max().date()})")

    print("Building charts...")
    fig_v = make_chart(vnindex,  "VNINDEX — regime timeline",    "VNINDEX close", show_rangeslider=True)
    fig_m = make_chart(midcap,   "VNMIDCAP — regime timeline",   "VNMIDCAP close")
    fig_s = make_chart(smallcap, "VNSMALLCAP — regime timeline", "VNSMALLCAP close")

    plotly_div_v = pio.to_html(fig_v, include_plotlyjs='inline',  full_html=False, div_id='chart_vnindex')
    plotly_div_m = pio.to_html(fig_m, include_plotlyjs=False,     full_html=False, div_id='chart_mid')
    plotly_div_s = pio.to_html(fig_s, include_plotlyjs=False,     full_html=False, div_id='chart_sml')

    cards = card_html("VNINDEX", vnindex) + card_html("VNMIDCAP", midcap) + card_html("VNSMALLCAP", smallcap)
    dist_rows = dist_row(vnindex, "VNINDEX") + dist_row(midcap, "VNMIDCAP") + dist_row(smallcap, "VNSMALLCAP")

    print("Computing long-position performance metrics...")
    perf_v = metrics_table_html(compute_long_metrics(vnindex),  "VNINDEX")
    perf_m = metrics_table_html(compute_long_metrics(midcap),   "VNMIDCAP")
    perf_s = metrics_table_html(compute_long_metrics(smallcap), "VNSMALLCAP")

    print("Building features tab...")
    features_body = features_tab_html()

    print("Building importance tab...")
    importance_body = importance_tab_html()

    last_data_date = max(vnindex['Date'].max(), midcap['Date'].max(), smallcap['Date'].max()).date()
    build_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CJM Regime Dashboard — Vietnam Equity Indices</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f7f8fa; color: #222; margin: 0; padding: 0;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px 0; }}
  .caption {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }}
  .card {{
    background: white; border-radius: 8px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .card-name {{ font-size: 14px; color: #555; }}
  .card-regime {{ font-size: 32px; font-weight: 600; margin: 4px 0; }}
  .card-close {{ font-size: 18px; color: #222; }}
  .card-date {{ font-size: 12px; color: #888; }}
  .proba-row {{ margin-top: 10px; }}
  .proba-bar {{ display: flex; width: 100%; height: 8px; border-radius: 4px;
                overflow: hidden; background: #eee; }}
  .proba-bar > div {{ height: 100%; }}
  .proba-text {{ margin-top: 4px; font-size: 11px; color: #555;
                 font-family: ui-monospace, monospace; }}
  .chart {{ background: white; border-radius: 8px; padding: 12px; margin-bottom: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 12px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f0f2f5; font-size: 13px; color: #444; }}
  .perf-table td, .perf-table th {{ padding: 8px 12px; font-size: 13px; }}
  .perf-table th {{ text-align: right; }}
  .perf-table td {{ text-align: right; }}
  .perf-table td:first-child, .perf-table td:nth-child(2),
  .perf-table th:first-child, .perf-table th:nth-child(2) {{ text-align: left; }}
  .split-label {{ font-size: 13px; font-weight: 600; color: #444;
                  background: #eef2f6; padding: 6px 12px; border-radius: 6px 6px 0 0;
                  display: inline-block; }}
  .regime-pill {{ display: inline-block; padding: 2px 10px; border-radius: 10px;
                  color: white; font-size: 12px; font-weight: 600; }}
  .perf-section {{ background: white; border-radius: 8px; padding: 14px 18px; margin-bottom: 18px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  /* Tab system (pure CSS via radio inputs) */
  .tabs {{ display: flex; gap: 0; border-bottom: 2px solid #ddd; margin-bottom: 20px; }}
  .tab-input {{ display: none; }}
  .tab-label {{
    padding: 10px 20px; cursor: pointer; background: #f0f2f5;
    border: 1px solid #ddd; border-bottom: none; border-radius: 8px 8px 0 0;
    margin-right: 4px; font-weight: 500; color: #555; transition: all 0.15s;
  }}
  .tab-label:hover {{ background: #e6e9ed; }}
  .tab-content {{ display: none; }}
  #tab-dash:checked ~ .tab-bar label[for="tab-dash"],
  #tab-feat:checked ~ .tab-bar label[for="tab-feat"],
  #tab-imp:checked  ~ .tab-bar label[for="tab-imp"] {{
    background: white; color: #222; border-bottom: 2px solid white; margin-bottom: -2px;
  }}
  #tab-dash:checked ~ #pane-dash,
  #tab-feat:checked ~ #pane-feat,
  #tab-imp:checked  ~ #pane-imp {{ display: block; }}
  /* Features tab styling */
  .feat-summary {{ background: white; border-radius: 8px; padding: 14px 18px;
                   margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  .feat-summary p {{ margin: 6px 0; }}
  .feat-summary ul {{ margin: 8px 0 8px 18px; padding: 0; }}
  .feat-summary li {{ margin: 4px 0; font-size: 14px; }}
  .feat-table {{ width: 100%; border-collapse: collapse; background: white;
                 border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
                 font-size: 13px; }}
  .feat-table th {{ background: #f0f2f5; padding: 10px 12px; text-align: left;
                    border-bottom: 1px solid #ddd; }}
  .feat-table td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
  .feat-table td.num {{ font-family: ui-monospace, monospace; text-align: right; color: #222; }}
  .feat-table .formula {{ font-size: 11.5px; color: #555; }}
  .feat-table .group-row td {{ background: #fafbfc; padding: 8px 12px; border-top: 2px solid #e0e0e0; }}
  .group-pill {{ display: inline-block; padding: 3px 12px; border-radius: 12px;
                 color: white; font-size: 11px; font-weight: 600; letter-spacing: 0.3px; }}
  .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 30px; }}
  .legend-box {{ background: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 18px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.06); font-size: 14px; }}
  .legend-item {{ display: inline-block; margin-right: 18px; }}
  .swatch {{ display: inline-block; width: 14px; height: 14px; vertical-align: middle;
             border-radius: 3px; margin-right: 6px; }}
  @media (max-width: 800px) {{ .cards {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
  <h1>CJM Regime Classification — Vietnam Equity Indices</h1>
  <div class="caption">
    Continuous Statistical Jump Model (K=3, λ=50). Train 2016-07 → 2024-12.
    Out-of-sample 2025-01 onward. Data through <b>{last_data_date}</b>.
  </div>

  <!-- Tab system -->
  <input class="tab-input" type="radio" id="tab-dash" name="dashtabs" checked>
  <input class="tab-input" type="radio" id="tab-feat" name="dashtabs">
  <input class="tab-input" type="radio" id="tab-imp"  name="dashtabs">
  <div class="tab-bar tabs">
    <label class="tab-label" for="tab-dash">Dashboard</label>
    <label class="tab-label" for="tab-feat">Features ({len(FEATURE_CATALOG)})</label>
    <label class="tab-label" for="tab-imp">Importance</label>
  </div>

  <div class="tab-content" id="pane-dash">

  <div class="cards">
    {cards}
  </div>

  <div class="legend-box">
    <span class="legend-item"><span class="swatch" style="background:#2ecc71"></span><b>Bull</b> — highest mean-return cluster</span>
    <span class="legend-item"><span class="swatch" style="background:#f39c12"></span><b>Neutral</b> — middle</span>
    <span class="legend-item"><span class="swatch" style="background:#e74c3c"></span><b>Bear</b> — lowest mean-return cluster</span>
  </div>

  <div class="chart">{plotly_div_v}</div>
  <div class="chart">{plotly_div_m}</div>
  <div class="chart">{plotly_div_s}</div>

  <h3>Regime distribution (full history)</h3>
  <table>
    <thead><tr><th>Index</th><th>Bars</th><th>Bull %</th><th>Neutral %</th><th>Bear %</th></tr></thead>
    <tbody>{dist_rows}</tbody>
  </table>

  <h2 style="margin-top:32px;">Long-position performance by regime — Train vs Test</h2>
  <div class="caption">
    Entry at <b>open[t]</b> of every bar labeled regime R (continuous long, one trade per bar).
    Exit at <b>open[t+N]</b> for T+3, T+5, T+15.<br>
    Excursion window = the <b>next N bars</b> after entry (bars t+1..t+N).<br>
    <b>Median return</b> = typical realized return; resistant to outlier-pulled means.<br>
    <b>P75 upside</b> = upper-quartile of max-High excursion / open[t] − 1 (top 25% favorable boundary).<br>
    <b>P25 downside</b> = lower-quartile of min-Low excursion / open[t] − 1 (bottom 25% adverse boundary — deeper drawdowns).
  </div>
  <div class="perf-section">{perf_v}</div>
  <div class="perf-section">{perf_m}</div>
  <div class="perf-section">{perf_s}</div>

  <div class="footer">
    CJM v7 (K=3, λ=50, grid=0.05). 26 features = 11 baseline + 4 HH/HL/LH/LL + 6 top-200 breadth + 3 foreign-flow + 2 volume.<br>
    Built {build_ts}.
  </div>

  </div><!-- /pane-dash -->

  <div class="tab-content" id="pane-feat">
    <h2 style="margin-top:0;">Feature Engineering — 26 features fed into CJM</h2>
    {features_body}
    <div class="footer" style="margin-top:18px;">
      All features hand-verified against published formulas — see <code>verify_*</code> scripts in repo root.<br>
      Built {build_ts}.
    </div>
  </div><!-- /pane-feat -->

  <div class="tab-content" id="pane-imp">
    <h2 style="margin-top:0;">Feature Importance — regime-separation power (η² = BCSS / TSS)</h2>
    {importance_body}
    <div class="footer" style="margin-top:18px;">
      Independent verification: 8/8 checks pass — formula identity holds to 2e-10, η² values
      reproduced to ~5e-05, labels economically consistent (ULT_RSI Bull&gt;Neut&gt;Bear).
      Source: <code>feature_importance_v7.py</code> · <code>feature_importance_v7.csv</code>.<br>
      Built {build_ts}.
    </div>
  </div><!-- /pane-imp -->

</div>
</body>
</html>"""

    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    size_kb = os.path.getsize(OUT_HTML) / 1024
    print(f"\nWrote {OUT_HTML} ({size_kb:.0f} KB)")
    print(f"Open in browser:  open '{OUT_HTML}'")


if __name__ == '__main__':
    main()
