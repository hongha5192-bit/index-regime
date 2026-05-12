# CJM Regime Dashboard — Vietnam Equity Indices

Continuous Statistical Jump Model (CJM v7, K=3, λ=50) for regime classification of VNINDEX / VNMIDCAP / VNSMALLCAP.
26 features per index: 11 baseline technicals + 4 streaks + 6 breadth + 3 foreign flow + 2 volume.
Train 2016-07 → 2024-12. Out-of-sample 2025-01 onward.

## Deploy to Streamlit Community Cloud

### 1. Push this folder to GitHub

```bash
cd webapp_deploy
git init
git add .
git commit -m "Initial CJM dashboard deploy"

# Create an empty repo on github.com (private or public), then:
git remote add origin git@github.com:<YOUR_USERNAME>/<REPO_NAME>.git
git branch -M main
git push -u origin main
```

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Repository: pick the repo you just pushed
5. Branch: `main`
6. Main file path: `streamlit_app.py`
7. Click **Deploy**

In ~2 minutes you get a permanent URL like
`https://<YOUR_USERNAME>-<REPO_NAME>.streamlit.app`

Share that link with friends.

## Local development

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## Layout

```
webapp_deploy/
├── streamlit_app.py          # entry point
├── build_dashboard.py        # data loaders + Plotly figure builders
├── data/                     # all data files (~4 MB total)
│   ├── phase4_cjm_v7_*.npz   # trained CJM regime labels
│   ├── phase4_regime_v7_*.csv# latest extended labels
│   ├── *_OHLCV_*.csv         # 26-feature CSVs per index
│   └── feature_importance_*.csv,png
├── requirements.txt
├── .streamlit/config.toml
└── .gitignore
```

## What you'll see

3 tabs:
- **Dashboard** — current regime per index, full timeline, distribution + performance metrics
- **Features (26)** — every input feature with formula, group, and latest snapshot value
- **Importance** — 4 metrics ranking each feature's regime-separation power
  - η² (univariate BCSS/TSS)
  - RF Gini (multivariate, TimeSeriesSplit CV)
  - SHAP per-class (XGBoost mimics CJM labels)
  - Wasserstein-1 (tail-sensitive)
  - Plus: side-by-side top-10 panels, SHAP heatmap, sortable full ranking table

## Caveats

- Streamlit Community Cloud apps sleep after inactivity — first visit may take ~30 sec to wake up
- Memory limit on free tier: 1 GB (we use ~250 MB so no issue)
- Repo must be GitHub (GitLab/Bitbucket not supported)
