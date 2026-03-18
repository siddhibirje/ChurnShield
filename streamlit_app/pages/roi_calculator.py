"""ROI Calculator — standalone Streamlit page."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
_sapp = os.path.join(ROOT, 'streamlit_app')
if _sapp not in sys.path: sys.path.insert(0, _sapp)

import streamlit as st
st.set_page_config(page_title="ROI Calculator | ChurnShield", page_icon="💰", layout="wide")

from utils.styles import apply_styles, sidebar_brand
apply_styles(); sidebar_brand()

import pandas as pd, numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_main_data, load_model, load_scaler, load_feature_names, models_trained

LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,.85)',
              font=dict(color='#94a3b8',family='Inter'), margin=dict(l=20,r=20,t=44,b=20),
              title_font=dict(color='#e2e8f0',size=15))

STRATEGIES = {
    "🎯 Premium: Manager + 20% Discount":{"cost":500,"lift":.40,"target":"High CLV",
        "desc":"Dedicated account manager + 20% discount for 6 months"},
    "💳 Standard: 15% Discount Campaign":{"cost":200,"lift":.30,"target":"Medium CLV",
        "desc":"Automated 15% personalised discount via email/SMS"},
    "📱 Basic: Loyalty Rewards + Upgrade":{"cost":100,"lift":.20,"target":"Low-medium risk",
        "desc":"Free service upgrade for 3 months + loyalty points"},
    "📧 Minimal: Email Re-engagement":{"cost":10,"lift":.10,"target":"Low CLV",
        "desc":"Automated email + push notification nurture sequence"},
}

st.markdown("# 💰 Retention ROI Calculator")
st.markdown("<p style='color:#94a3b8;margin-top:-10px;'>Simulate business impact of different retention campaigns in INR</p>",
            unsafe_allow_html=True)

df = load_main_data()

st.markdown("### ⚙️ Business Parameters")
p1, p2, p3 = st.columns(3)
with p1:
    margin = st.slider("Gross Margin (%)", 10, 60, 25) / 100
    inr    = st.slider("USD → INR Rate", 75, 95, 83)
with p2:
    thresh = st.slider("Flag at-risk above churn prob (%)", 30, 80, 50) / 100
    avg_life = st.slider("Avg Customer Lifetime (months)", 12, 60, 24)
with p3:
    strategy_name = st.selectbox("Retention Strategy", list(STRATEGIES.keys()))

strat = STRATEGIES[strategy_name]

# Get churn probabilities
if models_trained():
    mdl = load_model('best_model'); sc = load_scaler(); fn = load_feature_names()
    if mdl and sc and fn:
        drop = ['customerID','Churn','tenure_group','RFM_Segment','recency','frequency',
                'monetary','R_Score','F_Score','M_Score']
        Xr = df.drop([c for c in drop if c in df.columns], axis=1)
        Xe = pd.get_dummies(Xr).reindex(columns=fn, fill_value=0)
        df['churn_prob'] = mdl.predict_proba(sc.transform(Xe))[:,1]
    else:
        df['churn_prob'] = (1-(df['tenure']/df['tenure'].max()))*.6
else:
    df['churn_prob'] = (1-(df['tenure']/df['tenure'].max()))*.6

df['CLV'] = df['MonthlyCharges'] * avg_life * margin * inr
at_risk = df[df['churn_prob'] >= thresh].copy()
n = len(at_risk)
campaign_cost = n * strat['cost']
saved_n = n * strat['lift']
clv_saved = at_risk.nlargest(int(saved_n),'CLV')['CLV'].sum() if int(saved_n) > 0 else 0
net = clv_saved - campaign_cost
roi_pct = (net / campaign_cost * 100) if campaign_cost > 0 else 0
roi_color = '#22c55e' if roi_pct > 100 else '#f59e0b' if roi_pct > 0 else '#f43f5e'

st.markdown("---")
st.markdown("### 📊 Campaign Impact")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("At-Risk Customers", f"{n:,}", delta=f"{n/len(df):.1%} of base",delta_color="inverse")
k2.metric("Revenue at Risk", f"₹{at_risk['CLV'].sum():,.0f}")
k3.metric("Campaign Cost", f"₹{campaign_cost:,.0f}")
k4.metric("Est. Revenue Saved", f"₹{clv_saved:,.0f}")
k5.metric("Net ROI", f"{roi_pct:.0f}%", delta="vs. no action",delta_color="normal" if roi_pct>0 else "inverse")

st.markdown(f"""
<div class='card' style='border-top:3px solid {roi_color};'>
  <div style='font-size:1.1rem;font-weight:700;color:#a78bfa;margin-bottom:6px;'>{strategy_name}</div>
  <div style='color:#94a3b8;font-size:.85rem;margin-bottom:14px;'>{strat["desc"]}</div>
  <div style='display:flex;gap:36px;flex-wrap:wrap;'>
    <div><div style='color:#4a5568;font-size:.75rem;'>Cost per customer</div>
         <div style='color:#e2e8f0;font-weight:700;'>₹{strat["cost"]*inr:,.0f}</div></div>
    <div><div style='color:#4a5568;font-size:.75rem;'>Retention lift</div>
         <div style='color:#e2e8f0;font-weight:700;'>{strat["lift"]:.0%}</div></div>
    <div><div style='color:#4a5568;font-size:.75rem;'>Customers saved</div>
         <div style='color:#e2e8f0;font-weight:700;'>{saved_n:.0f}</div></div>
    <div><div style='color:#4a5568;font-size:.75rem;'>Net ROI</div>
         <div style='color:{roi_color};font-weight:800;font-size:1.3rem;'>{roi_pct:.0f}%</div></div>
  </div>
</div>""", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    fig_b = go.Figure(go.Bar(
        x=['Campaign Cost','Revenue Saved','Net Gain'],
        y=[campaign_cost, clv_saved, net],
        marker_color=['#f43f5e','#22c55e', roi_color],
        text=[f'₹{v:,.0f}' for v in [campaign_cost,clv_saved,net]],
        textposition='outside', textfont=dict(color='#e2e8f0')))
    fig_b.update_layout(**LAYOUT, title="Cost vs Revenue", height=320,
                        yaxis=dict(gridcolor='#1e2840'), xaxis=dict(gridcolor='#1e2840'))
    st.plotly_chart(fig_b, use_container_width=True)

with c2:
    fig_w = go.Figure(go.Waterfall(
        orientation='v', measure=['relative','relative','total'],
        x=['Revenue Saved','Campaign Cost','Net'],
        y=[clv_saved, -campaign_cost, 0],
        connector=dict(line=dict(color='#1e2840')),
        increasing=dict(marker=dict(color='#22c55e')),
        decreasing=dict(marker=dict(color='#f43f5e')),
        totals=dict(marker=dict(color=roi_color))))
    fig_w.update_layout(**LAYOUT, title="ROI Waterfall", height=320,
                        yaxis=dict(gridcolor='#1e2840'), xaxis=dict(gridcolor='#1e2840'))
    st.plotly_chart(fig_w, use_container_width=True)

st.markdown("### 🔄 All Strategies Compared")
cmp = []
for sn, s in STRATEGIES.items():
    cc = n * s['cost']
    sv = at_risk.nlargest(int(n*s['lift']),'CLV')['CLV'].sum() if int(n*s['lift']) > 0 else 0
    cmp.append({'Strategy':sn.split(":")[0].strip(),'Cost (₹)':int(cc),
                'Revenue Saved (₹)':int(sv),'ROI (%)':round((sv-cc)/cc*100,1) if cc>0 else 0,
                'Customers Saved':int(n*s['lift'])})
cmp_df = pd.DataFrame(cmp)
st.dataframe(cmp_df.set_index('Strategy'), use_container_width=True)

fig_cmp = go.Figure(go.Bar(
    x=cmp_df['Strategy'], y=cmp_df['ROI (%)'],
    marker_color=['#22c55e' if v>100 else '#f59e0b' if v>0 else '#f43f5e' for v in cmp_df['ROI (%)']],
    text=cmp_df['ROI (%)'].apply(lambda x:f"{x:.0f}%"), textposition='outside'))
fig_cmp.update_layout(**LAYOUT, title="ROI by Strategy", height=300,
                      yaxis=dict(title='ROI (%)',gridcolor='#1e2840'),
                      xaxis=dict(gridcolor='#1e2840'), showlegend=False)
st.plotly_chart(fig_cmp, use_container_width=True)
