"""Model Performance page — standalone Streamlit page."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
_sapp = os.path.join(ROOT, 'streamlit_app')
if _sapp not in sys.path: sys.path.insert(0, _sapp)

import streamlit as st
st.set_page_config(page_title="Model Performance | ChurnShield", page_icon="📈", layout="wide")

from utils.styles import apply_styles, sidebar_brand
apply_styles(); sidebar_brand()

import pandas as pd, numpy as np
import plotly.graph_objects as go, plotly.figure_factory as ff
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.data_loader import load_model, load_test_data, load_model_comparison, load_feature_names, models_trained

LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,.85)',
              font=dict(color='#94a3b8', family='Inter'), margin=dict(l=20,r=20,t=44,b=20),
              title_font=dict(color='#e2e8f0', size=15))
COLORS = ['#6366f1','#f43f5e','#22c55e','#f59e0b','#06b6d4','#a78bfa']
MODEL_FILES = {'Logistic Regression': 'logistic_regression',
               'Decision Tree': 'decision_tree',
               'Random Forest': 'random_forest',
               'XGBoost': 'xgboost',
               'LightGBM': 'lightgbm',
               'Voting Ensemble': 'voting_ensemble'}

# ── Page Header ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#e2e8f0;font-size:2rem;margin-bottom:4px;'>📈 Model Performance</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;margin-top:0;margin-bottom:24px;'>Compare all trained classifiers side-by-side</p>",
    unsafe_allow_html=True)

if not models_trained():
    st.warning("⚠️ Models not trained yet. Run `python train_models.py` first.")
    st.code("python train_models.py")
    st.stop()

td = load_test_data()
if td is None:
    st.error("Test data missing. Re-run `python train_models.py`.")
    st.stop()
X_test, y_test = td['X_test'], td['y_test']

# ── Model Comparison Table ───────────────────────────────────────────────────
comp = load_model_comparison()
if comp is not None:
    st.markdown(
        "<h3 style='color:#e2e8f0;'>📋 Model Comparison</h3>",
        unsafe_allow_html=True)

    # Rename columns
    col_map = {
        'auc_roc': 'AUC-ROC',
        'avg_precision': 'Avg Precision',
        'accuracy': 'Accuracy',
        'precision_churn': 'Precision (Churn)',
        'recall_churn': 'Recall (Churn)',
        'f1_churn': 'F1 (Churn)'
    }
    disp = comp.rename(columns=col_map)
    disp = disp.apply(pd.to_numeric, errors='coerce').round(4)

    numeric_cols = [c for c in ['AUC-ROC','Avg Precision','Accuracy','F1 (Churn)'] if c in disp.columns]

    # Find best value per numeric column to highlight
    disp_fmt = disp.copy()
    cell_colors = []
    for col in disp_fmt.columns:
        col_colors = []
        for val in disp_fmt[col]:
            if col in numeric_cols and pd.notna(val) and val == disp_fmt[col].max():
                col_colors.append('rgba(99,102,241,0.45)')
            else:
                col_colors.append('rgba(17,24,39,0.85)')
        cell_colors.append(col_colors)

    header_vals = ['Model'] + list(disp_fmt.columns)
    cell_vals   = [list(disp_fmt.index)] + [
        [f"{v:.4f}" if pd.notna(v) else "-" for v in disp_fmt[c]]
        for c in disp_fmt.columns
    ]
    cell_fill   = [['rgba(30,40,64,0.9)'] * len(disp_fmt)] + cell_colors

    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=header_vals,
            fill_color='#1e2840',
            font=dict(color='#e2e8f0', size=13, family='Inter'),
            align='left',
            height=36,
            line=dict(color='#2d3748', width=1)
        ),
        cells=dict(
            values=cell_vals,
            fill_color=cell_fill,
            font=dict(color='#cbd5e1', size=12, family='Inter'),
            align='left',
            height=32,
            line=dict(color='#2d3748', width=1)
        )
    ))
    fig_tbl.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=40 + 36 + 32 * len(disp_fmt)
    )
    st.plotly_chart(fig_tbl, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
else:
    st.info("Model comparison CSV not found. Re-run `python train_models.py`.")

# ── Load Models ──────────────────────────────────────────────────────────────
loaded = {}
for name, fname in MODEL_FILES.items():
    m = load_model(fname)
    if m:
        loaded[name] = m

if not loaded:
    st.error("No models found.")
    st.stop()

# ── ROC Curves ───────────────────────────────────────────────────────────────
st.markdown("<h3 style='color:#e2e8f0;'>📉 ROC Curves</h3>", unsafe_allow_html=True)
fig_roc = go.Figure()
fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1,
                  line=dict(color='#4a5568', dash='dash', width=1))
for i, (name, mdl) in enumerate(loaded.items()):
    try:
        prob = mdl.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC={auc(fpr,tpr):.3f})',
            mode='lines',
            line=dict(color=COLORS[i % len(COLORS)], width=2.5)))
    except Exception:
        pass
fig_roc.update_layout(**LAYOUT, height=420,
                      xaxis=dict(title='False Positive Rate', gridcolor='#1e2840', range=[0,1]),
                      yaxis=dict(title='True Positive Rate', gridcolor='#1e2840', range=[0,1]),
                      legend=dict(bgcolor='rgba(0,0,0,0)', x=.55, y=.1))
st.plotly_chart(fig_roc, use_container_width=True)

# ── PR Curves + Confusion Matrix ─────────────────────────────────────────────
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("<h3 style='color:#e2e8f0;'>🎯 Precision-Recall Curves</h3>", unsafe_allow_html=True)
    fig_pr = go.Figure()
    for i, (name, mdl) in enumerate(loaded.items()):
        try:
            pr, rc, _ = precision_recall_curve(y_test, mdl.predict_proba(X_test)[:, 1])
            fig_pr.add_trace(go.Scatter(x=rc, y=pr, name=name, mode='lines',
                                        line=dict(color=COLORS[i % len(COLORS)], width=2)))
        except Exception:
            pass
    fig_pr.update_layout(**LAYOUT, height=360,
                         xaxis=dict(title='Recall', gridcolor='#1e2840'),
                         yaxis=dict(title='Precision', gridcolor='#1e2840'),
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_pr, use_container_width=True)

with c2:
    st.markdown("<h3 style='color:#e2e8f0;'>🧩 Confusion Matrix</h3>", unsafe_allow_html=True)
    sel = st.selectbox("Select model", list(loaded.keys()), label_visibility="collapsed")
    preds = loaded[sel].predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=['Retained', 'Churned'],
        y=['Retained', 'Churned'],
        colorscale=[[0,'#111827'],[1,'#6366f1']],
        annotation_text=[[str(v) for v in r] for r in cm],
        showscale=False)
    fig_cm.update_layout(**LAYOUT, height=320,
                         xaxis=dict(title='Predicted'),
                         yaxis=dict(title='Actual', autorange='reversed'))
    st.plotly_chart(fig_cm, use_container_width=True)

# ── Feature Importance ───────────────────────────────────────────────────────
st.markdown("<h3 style='color:#e2e8f0;'>🔍 Feature Importance</h3>", unsafe_allow_html=True)
st.caption("Based on Random Forest / XGBoost / LightGBM — whichever is available.")
feat_names = load_feature_names()
fi_mdl = loaded.get('Random Forest') or loaded.get('XGBoost') or loaded.get('LightGBM')
if fi_mdl and feat_names and hasattr(fi_mdl, 'feature_importances_'):
    fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': fi_mdl.feature_importances_})
    fi_df = fi_df.sort_values('Importance', ascending=True).tail(20)
    fig_fi = go.Figure(go.Bar(
        x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
        marker=dict(color=fi_df['Importance'],
                    colorscale=[[0,'#1e2840'],[.5,'#6366f1'],[1,'#a78bfa']])))
    fig_fi.update_layout(**LAYOUT, height=520, title="Top 20 Features by Importance",
                         xaxis=dict(title='Importance', gridcolor='#1e2840'),
                         yaxis=dict(gridcolor='#1e2840'))
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("Feature importance not available for this model type.")

# ── Threshold Optimizer ──────────────────────────────────────────────────────
st.markdown("<h3 style='color:#e2e8f0;'>⚖️ Threshold Optimizer</h3>", unsafe_allow_html=True)
st.caption("Lower threshold = catch more churners (higher recall). Higher = fewer false alarms (higher precision).")

best_mdl = loaded.get('Random Forest') or list(loaded.values())[0]
prob_t = best_mdl.predict_proba(X_test)[:, 1]

thresh = st.slider("Classification Threshold", 0.10, 0.90, 0.50, 0.05,
                   help="Default 0.50 — move left to flag more customers as churners")
preds_t = (prob_t >= thresh).astype(int)

prec  = precision_score(y_test, preds_t, zero_division=0)
rec   = recall_score(y_test, preds_t, zero_division=0)
f1    = f1_score(y_test, preds_t, zero_division=0)
flagged = int(preds_t.sum())
total   = len(y_test)

tc1, tc2, tc3, tc4 = st.columns(4)
tc1.metric("Precision",          f"{prec:.1%}",
           help="Of all flagged churners, how many actually churn?")
tc2.metric("Recall",             f"{rec:.1%}",
           help="Of all actual churners, how many did we catch?")
tc3.metric("F1 Score",           f"{f1:.1%}",
           help="Harmonic mean of precision and recall")
tc4.metric("Flagged as Churners", f"{flagged:,} / {total:,}",
           help="Number of customers flagged at this threshold")

# Visual tradeoff bar
st.markdown("<br>", unsafe_allow_html=True)
fig_thresh = go.Figure()

thresholds = np.arange(0.10, 0.91, 0.05)
precs, recs, f1s = [], [], []
for t in thresholds:
    p_t = (prob_t >= t).astype(int)
    precs.append(precision_score(y_test, p_t, zero_division=0))
    recs.append(recall_score(y_test, p_t, zero_division=0))
    f1s.append(f1_score(y_test, p_t, zero_division=0))

fig_thresh.add_trace(go.Scatter(x=thresholds, y=precs, name='Precision',
                                 mode='lines+markers', line=dict(color='#6366f1', width=2)))
fig_thresh.add_trace(go.Scatter(x=thresholds, y=recs, name='Recall',
                                 mode='lines+markers', line=dict(color='#f43f5e', width=2)))
fig_thresh.add_trace(go.Scatter(x=thresholds, y=f1s, name='F1 Score',
                                 mode='lines+markers', line=dict(color='#22c55e', width=2)))
fig_thresh.add_vline(x=thresh, line=dict(color='#f59e0b', dash='dash', width=1.5),
                     annotation_text=f"Current: {thresh}", annotation_font_color='#f59e0b')
fig_thresh.update_layout(**LAYOUT, height=340,
                          title="Precision / Recall / F1 across thresholds",
                          xaxis=dict(title='Threshold', gridcolor='#1e2840', tickformat='.2f'),
                          yaxis=dict(title='Score', gridcolor='#1e2840', range=[0,1]),
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
st.plotly_chart(fig_thresh, use_container_width=True)