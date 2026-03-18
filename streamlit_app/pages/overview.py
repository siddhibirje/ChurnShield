"""Overview page — runs standalone via Streamlit multi-page routing."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import streamlit as st
st.set_page_config(page_title="Overview | ChurnShield", page_icon="🏠", layout="wide")

# load shared styles
_sapp = os.path.join(ROOT, 'streamlit_app')
if _sapp not in sys.path: sys.path.insert(0, _sapp)
from utils.styles import apply_styles, sidebar_brand
apply_styles()
sidebar_brand()

import pandas as pd, numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_main_data, models_trained, data_available

LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,.85)',
              font=dict(color='#94a3b8',family='Inter'), margin=dict(l=20,r=20,t=44,b=20),
              title_font=dict(color='#e2e8f0',size=15))

st.markdown("# 🏠 Overview")
st.markdown("<p style='color:#94a3b8;margin-top:-10px;'>Real-time churn intelligence across your subscriber base</p>",
            unsafe_allow_html=True)

if not data_available():
    st.error("⚠️ No dataset found. Run `python generate_sample_data.py` first.")
    st.code("python generate_sample_data.py\npython train_models.py", language="bash")
    st.stop()

df = load_main_data()
if not models_trained():
    st.warning("⚠️ Models not trained yet — showing EDA only. Run `python train_models.py`.")

total = len(df); churned = int(df['Churn'].sum()); retained = total - churned
churn_rate = churned / total; avg_monthly = df['MonthlyCharges'].mean()

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Churned", f"{churned:,}", delta=f"-{churn_rate:.1%}", delta_color="inverse")
c3.metric("Retained", f"{retained:,}", delta=f"+{1-churn_rate:.1%}")
c4.metric("Avg Monthly (₹)", f"₹{avg_monthly*83:.0f}")
c5.metric("Avg Tenure", f"{df['tenure'].mean():.1f} mo")

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([1,2], gap="large")

with col1:
    fig = go.Figure(go.Pie(
        labels=['Retained','Churned'], values=[retained, churned], hole=.65,
        marker=dict(colors=['#6366f1','#f43f5e'], line=dict(color='#0a0d14',width=3)),
        textinfo='percent', textfont=dict(color='#e2e8f0',size=13)))
    fig.add_annotation(text=f"<b>{churn_rate:.1%}</b><br><span style='font-size:11px'>Churn Rate</span>",
                       x=.5,y=.5,showarrow=False,font=dict(size=20,color='#f43f5e'))
    fig.update_layout(**LAYOUT, title="Churn Distribution", height=300, showlegend=True,
                      legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    cc = df.groupby('Contract')['Churn'].mean().reset_index()
    cc['pct'] = (cc['Churn']*100).round(1)
    fig2 = go.Figure(go.Bar(
        x=cc['Contract'], y=cc['pct'],
        marker_color=['#f43f5e','#f59e0b','#22c55e'],
        text=cc['pct'].apply(lambda x:f"{x:.1f}%"), textposition='outside',
        textfont=dict(color='#e2e8f0')))
    fig2.update_layout(**LAYOUT, title="Churn Rate by Contract Type", height=300,
                       yaxis=dict(title='Churn %', gridcolor='#1e2840'),
                       xaxis=dict(gridcolor='#1e2840'))
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2, gap="large")
with col3:
    fig3 = go.Figure()
    for v, lbl, col in [(0,'Retained','#6366f1'),(1,'Churned','#f43f5e')]:
        fig3.add_trace(go.Histogram(x=df[df['Churn']==v]['tenure'], name=lbl,
                                    marker_color=col, opacity=.75, nbinsx=30))
    fig3.update_layout(**LAYOUT, title="Tenure Distribution", barmode='overlay', height=300,
                       xaxis=dict(title='Months', gridcolor='#1e2840'),
                       yaxis=dict(title='Count', gridcolor='#1e2840'),
                       legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = go.Figure()
    for v, lbl, col in [(0,'Retained','#6366f1'),(1,'Churned','#f43f5e')]:
        fig4.add_trace(go.Box(y=df[df['Churn']==v]['MonthlyCharges'], name=lbl,
                              marker_color=col, line_color=col))
    fig4.update_layout(**LAYOUT, title="Monthly Charges by Churn", height=300,
                       yaxis=dict(title='$/mo', gridcolor='#1e2840'),
                       legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("### 🔥 Churn Heatmap — Internet × Contract")
pivot = df.pivot_table(values='Churn', index='InternetService', columns='Contract', aggfunc='mean') * 100
fig5 = go.Figure(go.Heatmap(
    z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
    colorscale=[[0,'#0f1520'],[.5,'#6366f1'],[1,'#f43f5e']],
    text=pivot.values.round(1), texttemplate='%{text}%',
    textfont=dict(color='white',size=15), showscale=True))
fig5.update_layout(**LAYOUT, height=280,
                   xaxis=dict(title='Contract Type'), yaxis=dict(title='Internet Service'))
st.plotly_chart(fig5, use_container_width=True)

st.markdown("### 💳 Churn by Payment Method")
pay = df.groupby('PaymentMethod')['Churn'].mean().reset_index()
pay['pct'] = (pay['Churn']*100).round(1)
pay = pay.sort_values('pct', ascending=True)
fig6 = go.Figure(go.Bar(
    y=pay['PaymentMethod'], x=pay['pct'], orientation='h',
    marker=dict(color=pay['pct'],
                colorscale=[[0,'#22c55e'],[.5,'#f59e0b'],[1,'#f43f5e']]),
    text=pay['pct'].apply(lambda x:f"{x:.1f}%"), textposition='outside',
    textfont=dict(color='#e2e8f0')))
fig6.update_layout(**LAYOUT, height=250,
                   xaxis=dict(title='Churn %', gridcolor='#1e2840'),
                   yaxis=dict(gridcolor='#1e2840'))
st.plotly_chart(fig6, use_container_width=True)
