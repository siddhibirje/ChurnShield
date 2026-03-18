"""Segments page — RFM + K-Means clustering. Standalone Streamlit page."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
_sapp = os.path.join(ROOT, 'streamlit_app')
if _sapp not in sys.path: sys.path.insert(0, _sapp)

import streamlit as st
st.set_page_config(page_title="Segments | ChurnShield", page_icon="👥", layout="wide")

from utils.styles import apply_styles, sidebar_brand
apply_styles(); sidebar_brand()

import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from utils.data_loader import load_main_data

LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,.85)',
              font=dict(color='#94a3b8',family='Inter'), margin=dict(l=20,r=20,t=44,b=20),
              title_font=dict(color='#e2e8f0',size=15))
COLORS = ['#6366f1','#f43f5e','#22c55e','#f59e0b','#06b6d4']
SEG_COLORS = {'Champions':'#22c55e','Loyal Customers':'#6366f1',
              'Potential Loyalists':'#06b6d4','At Risk':'#f59e0b','Lost':'#f43f5e'}

st.markdown("# 👥 Customer Segments")
st.markdown("<p style='color:#94a3b8;margin-top:-10px;'>RFM lifecycle segments and K-Means behavioural clusters</p>",
            unsafe_allow_html=True)

df = load_main_data()
tab1, tab2 = st.tabs(["📊  RFM Segments", "🔵  Behavioural Clusters"])

# ── RFM TAB ──────────────────────────────────────────────────────────────────
with tab1:
    if 'RFM_Segment' not in df.columns:
        st.error("RFM features not found. Run feature engineering first."); st.stop()

    summary = df.groupby('RFM_Segment').agg(
        Customers=('customerID','count'), Churn_Rate=('Churn','mean'),
        Avg_Revenue=('MonthlyCharges','mean'), Avg_Tenure=('tenure','mean')).reset_index()
    summary['Churn_%'] = (summary['Churn_Rate']*100).round(1)
    summary['Avg_Revenue'] = summary['Avg_Revenue'].round(2)

    # Segment cards
    cols = st.columns(len(summary))
    for i, row in summary.iterrows():
        color = SEG_COLORS.get(row['RFM_Segment'],'#6366f1')
        cols[i % len(cols)].markdown(f"""
        <div class='card' style='border-top:3px solid {color};text-align:center;'>
          <div style='color:{color};font-weight:700;font-size:.9rem;'>{row['RFM_Segment']}</div>
          <div style='font-size:1.8rem;font-weight:800;color:#e2e8f0;'>{int(row['Customers']):,}</div>
          <div style='color:#94a3b8;font-size:.75rem;'>customers</div>
          <hr style='border-color:#1e2840;margin:8px 0;'>
          <div style='color:#f43f5e;font-weight:700;'>{row['Churn_%']}% churn</div>
          <div style='color:#94a3b8;font-size:.75rem;'>₹{row['Avg_Revenue']*83:.0f}/mo avg</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_b = go.Figure()
        for _, row in summary.iterrows():
            color = SEG_COLORS.get(row['RFM_Segment'],'#6366f1')
            fig_b.add_trace(go.Scatter(
                x=[row['Customers']], y=[row['Churn_%']],
                mode='markers+text',
                marker=dict(size=row['Avg_Revenue']*1.4, color=color, opacity=.8,
                            line=dict(width=1,color='white')),
                text=[row['RFM_Segment']], textposition='top center', name=row['RFM_Segment']))
        fig_b.update_layout(**LAYOUT, title="Size vs Churn (bubble=revenue)", height=320,
                            showlegend=False,
                            xaxis=dict(title='Customers',gridcolor='#1e2840'),
                            yaxis=dict(title='Churn %',gridcolor='#1e2840'))
        st.plotly_chart(fig_b, use_container_width=True)

    with c2:
        cats = ['R_Score','F_Score','M_Score']
        if all(c in df.columns for c in cats):
            means = df.groupby('RFM_Segment')[cats].mean()
            fig_r = go.Figure()
            for seg, row in means.iterrows():
                col = SEG_COLORS.get(seg,'#6366f1')
                vals = list(row)+[row[0]]
                fig_r.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]],
                                                 name=seg, fill='toself', opacity=.6,
                                                 line=dict(color=col)))
            fig_r.update_layout(**LAYOUT, title="RFM Radar by Segment", height=320,
                                polar=dict(bgcolor='rgba(17,24,39,.85)',
                                           radialaxis=dict(range=[0,5],gridcolor='#1e2840'),
                                           angularaxis=dict(gridcolor='#1e2840')),
                                legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig_r, use_container_width=True)

# ── CLUSTER TAB ───────────────────────────────────────────────────────────────
with tab2:
    k = st.slider("Number of clusters (K)", 2, 6, 4)
    feat_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges','engagement_score',
                              'charges_per_month','SeniorCitizen'] if c in df.columns]
    X = df[feat_cols].fillna(0)
    Xs = StandardScaler().fit_transform(X)

    ins, sils = [], []
    for ki in range(2,9):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        lbs = km.fit_predict(Xs); ins.append(km.inertia_)
        sils.append(silhouette_score(Xs, lbs))

    ec1, ec2 = st.columns(2)
    with ec1:
        fig_e = go.Figure(go.Scatter(x=list(range(2,9)), y=ins, mode='lines+markers',
                                      marker=dict(color='#6366f1',size=8), line=dict(color='#6366f1',width=2)))
        fig_e.add_vline(x=k, line=dict(color='#f43f5e',dash='dash'))
        fig_e.update_layout(**LAYOUT, title="Elbow Curve", height=260,
                            xaxis=dict(title='K',gridcolor='#1e2840'),
                            yaxis=dict(title='Inertia',gridcolor='#1e2840'))
        st.plotly_chart(fig_e, use_container_width=True)

    with ec2:
        fig_s = go.Figure(go.Scatter(x=list(range(2,9)), y=sils, mode='lines+markers',
                                      marker=dict(color='#22c55e',size=8), line=dict(color='#22c55e',width=2)))
        fig_s.add_vline(x=k, line=dict(color='#f43f5e',dash='dash'))
        fig_s.update_layout(**LAYOUT, title="Silhouette Score", height=260,
                            xaxis=dict(title='K',gridcolor='#1e2840'),
                            yaxis=dict(title='Score',gridcolor='#1e2840'))
        st.plotly_chart(fig_s, use_container_width=True)

    km_f = KMeans(n_clusters=k, random_state=42, n_init=10)
    df2 = df.copy(); df2['Cluster'] = km_f.fit_predict(Xs).astype(str)
    coords = PCA(n_components=2).fit_transform(Xs)
    pca_df = pd.DataFrame(coords, columns=['PC1','PC2'])
    pca_df['Cluster'] = df2['Cluster'].values
    pca_df['Churn'] = df2['Churn'].map({1:'Churned',0:'Retained'})

    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', symbol='Churn',
                         color_discrete_map={str(i):COLORS[i%len(COLORS)] for i in range(k)},
                         title=f'K-Means PCA Scatter (K={k})', opacity=.7)
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(**LAYOUT, height=380,
                          xaxis=dict(gridcolor='#1e2840'), yaxis=dict(gridcolor='#1e2840'),
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("### 🔬 Cluster Profiles")
    prof = df2.groupby('Cluster').agg(
        Count=('customerID','count'), Churn_Rate=('Churn','mean'),
        Avg_Tenure=('tenure','mean'), Avg_Monthly=('MonthlyCharges','mean')).round(2)
    prof['Churn_Rate'] = (prof['Churn_Rate']*100).round(1)
    prof.columns = ['Count','Churn Rate (%)','Avg Tenure (mo)','Avg Monthly ($)']
    st.dataframe(prof, use_container_width=True)
