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
              font=dict(color='#94a3b8',family='Inter'), margin=dict(l=20,r=20,t=44,b=20),
              title_font=dict(color='#e2e8f0',size=15))
COLORS = ['#6366f1','#f43f5e','#22c55e','#f59e0b','#06b6d4','#a78bfa']
MODEL_FILES = {'Logistic Regression':'logistic_regression','Decision Tree':'decision_tree',
               'Random Forest':'random_forest','XGBoost':'xgboost',
               'LightGBM':'lightgbm','Voting Ensemble':'voting_ensemble'}

st.markdown("# 📈 Model Performance")
st.markdown("<p style='color:#94a3b8;margin-top:-10px;'>Compare all trained classifiers side-by-side</p>",
            unsafe_allow_html=True)

if not models_trained():
    st.warning("⚠️ Models not trained yet. Run `python train_models.py` first.")
    st.code("python train_models.py"); st.stop()

td = load_test_data()
if td is None:
    st.error("Test data missing. Re-run `python train_models.py`."); st.stop()
X_test, y_test = td['X_test'], td['y_test']

comp = load_model_comparison()
if comp is not None:
    st.markdown("### 📋 Model Comparison")
    disp = comp.rename(columns={'auc_roc':'AUC-ROC','avg_precision':'Avg Precision',
                                 'accuracy':'Accuracy','precision_churn':'Precision (Churn)',
                                 'recall_churn':'Recall (Churn)','f1_churn':'F1 (Churn)'})
    st.dataframe(disp.style.highlight_max(
        subset=['AUC-ROC','Avg Precision','Accuracy','F1 (Churn)'],
        color='rgba(99,102,241,.35)').format("{:.4f}"), use_container_width=True)

loaded = {}
for name, fname in MODEL_FILES.items():
    m = load_model(fname)
    if m: loaded[name] = m

if not loaded:
    st.error("No models found."); st.stop()

# ROC curves
st.markdown("### 📉 ROC Curves")
fig_roc = go.Figure()
fig_roc.add_shape(type='line', x0=0,x1=1,y0=0,y1=1,line=dict(color='#4a5568',dash='dash',width=1))
for i, (name, mdl) in enumerate(loaded.items()):
    try:
        prob = mdl.predict_proba(X_test)[:,1]
        fpr,tpr,_ = roc_curve(y_test,prob)
        fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,name=f'{name} (AUC={auc(fpr,tpr):.3f})',
                                     mode='lines',line=dict(color=COLORS[i%len(COLORS)],width=2.5)))
    except Exception: pass
fig_roc.update_layout(**LAYOUT, height=400,
                      xaxis=dict(title='False Positive Rate',gridcolor='#1e2840',range=[0,1]),
                      yaxis=dict(title='True Positive Rate',gridcolor='#1e2840',range=[0,1]),
                      legend=dict(bgcolor='rgba(0,0,0,0)',x=.55,y=.1))
st.plotly_chart(fig_roc, use_container_width=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("### 🎯 Precision-Recall Curves")
    fig_pr = go.Figure()
    for i, (name, mdl) in enumerate(loaded.items()):
        try:
            pr,rc,_ = precision_recall_curve(y_test, mdl.predict_proba(X_test)[:,1])
            fig_pr.add_trace(go.Scatter(x=rc,y=pr,name=name,mode='lines',
                                        line=dict(color=COLORS[i%len(COLORS)],width=2)))
        except Exception: pass
    fig_pr.update_layout(**LAYOUT, height=340,
                         xaxis=dict(title='Recall',gridcolor='#1e2840'),
                         yaxis=dict(title='Precision',gridcolor='#1e2840'),
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_pr, use_container_width=True)

with c2:
    st.markdown("### 🧩 Confusion Matrix")
    sel = st.selectbox("Model", list(loaded.keys()), label_visibility="collapsed")
    preds = loaded[sel].predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig_cm = ff.create_annotated_heatmap(z=cm, x=['Retained','Churned'], y=['Retained','Churned'],
                                          colorscale=[[0,'#111827'],[1,'#6366f1']],
                                          annotation_text=[[str(v) for v in r] for r in cm],
                                          showscale=False)
    fig_cm.update_layout(**LAYOUT, height=300,
                         xaxis=dict(title='Predicted'),
                         yaxis=dict(title='Actual',autorange='reversed'))
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("### 🔍 Feature Importance (Random Forest / XGBoost / LightGBM)")
feat_names = load_feature_names()
fi_mdl = loaded.get('Random Forest') or loaded.get('XGBoost') or loaded.get('LightGBM')
if fi_mdl and feat_names and hasattr(fi_mdl,'feature_importances_'):
    fi_df = pd.DataFrame({'Feature':feat_names,'Importance':fi_mdl.feature_importances_})
    fi_df = fi_df.sort_values('Importance',ascending=True).tail(20)
    fig_fi = go.Figure(go.Bar(x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
                               marker=dict(color=fi_df['Importance'],
                                           colorscale=[[0,'#1e2840'],[.5,'#6366f1'],[1,'#a78bfa']])))
    fig_fi.update_layout(**LAYOUT, height=500, title="Top 20 Features",
                         xaxis=dict(title='Importance',gridcolor='#1e2840'),
                         yaxis=dict(gridcolor='#1e2840'))
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("### ⚖️ Threshold Optimizer")
st.caption("Lower threshold = catch more churners (higher recall), fewer false negatives. Adjust for business needs.")
thresh = st.slider("Classification Threshold", .10, .90, .50, .05)
best_mdl = loaded.get('Random Forest') or list(loaded.values())[0]
prob_t = best_mdl.predict_proba(X_test)[:,1]
preds_t = (prob_t >= thresh).astype(int)
tc1,tc2,tc3,tc4 = st.columns(4)
tc1.metric("Precision", f"{precision_score(y_test,preds_t,zero_division=0):.1%}")
tc2.metric("Recall", f"{recall_score(y_test,preds_t,zero_division=0):.1%}")
tc3.metric("F1 Score", f"{f1_score(y_test,preds_t,zero_division=0):.1%}")
tc4.metric("Flagged as Churners", f"{int(preds_t.sum()):,}")
