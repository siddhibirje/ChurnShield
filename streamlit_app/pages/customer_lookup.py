"""Customer Lookup page — standalone Streamlit page."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
_sapp = os.path.join(ROOT, 'streamlit_app')
if _sapp not in sys.path: sys.path.insert(0, _sapp)

import streamlit as st
st.set_page_config(page_title="Customer Lookup | ChurnShield", page_icon="🔍", layout="wide")

from utils.styles import apply_styles, sidebar_brand
apply_styles()
sidebar_brand()

import pandas as pd, numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_main_data, load_model, load_scaler, load_feature_names, models_trained

def get_risk(prob):
    if prob >= .65: return "HIGH RISK", "badge-high", "🔴"
    elif prob >= .35: return "MEDIUM RISK", "badge-med", "🟡"
    return "LOW RISK", "badge-low", "🟢"

def strategy(prob, clv):
    if prob >= .65 and clv >= 500:
        return "🎯 Personal Manager + 20% Discount", "Immediate personal outreach. High-value at critical risk.", "₹500", "40%"
    elif prob >= .65:
        return "📧 Targeted 15% Discount", "Send personalised discount via email/SMS within 24h.", "₹200", "30%"
    elif prob >= .35 and clv >= 300:
        return "🎁 Loyalty Rewards + Upgrade Offer", "Reward tenure with a free service upgrade for 3 months.", "₹300", "20%"
    elif prob >= .35:
        return "📱 Automated Re-engagement", "Email + push notification nurture sequence.", "₹50", "15%"
    return "✅ Monitor — Low Risk", "Continue standard experience. Review in 3 months.", "₹10", "5%"

st.markdown("# 🔍 Customer Lookup")
st.markdown("<p style='color:#94a3b8;margin-top:-10px;'>Search any customer for their individual churn risk & recommended action</p>",
            unsafe_allow_html=True)

if not models_trained():
    st.warning("⚠️ Models not trained. Run `python train_models.py` first.")
    st.stop()

df = load_main_data()
model = load_model('best_model')
scaler = load_scaler()
feature_names = load_feature_names()

if model is None or scaler is None or feature_names is None:
    st.error("Model files missing. Run `python train_models.py`.")
    st.stop()

c1, c2 = st.columns([4,1])
with c1:
    query = st.text_input("", placeholder="🔎  Enter Customer ID  (e.g. CUST-00042)", label_visibility="collapsed")
with c2:
    if st.button("🎲 Random"):
        st.session_state.rand_cid = np.random.choice(df['customerID'])

if 'rand_cid' in st.session_state and not query:
    query = st.session_state.rand_cid

if not query:
    st.markdown("### 💡 Sample Customer IDs to try:")
    cols = st.columns(6)
    for i, cid in enumerate(df.sample(6)['customerID']):
        cols[i].code(cid)
    st.stop()

match = df[df['customerID'].str.lower() == query.strip().lower()]
if match.empty:
    match = df[df['customerID'].str.contains(query.strip(), case=False, na=False)]
if match.empty:
    st.error(f"No customer found for '{query}'")
    st.stop()

customer = match.iloc[0]
st.success(f"✅  Found: **{customer['customerID']}**")

drop_cols = ['customerID','Churn','tenure_group','RFM_Segment','recency','frequency','monetary','R_Score','F_Score','M_Score']
row = customer.drop([c for c in drop_cols if c in customer.index], errors='ignore')
row_df = pd.DataFrame([row])
row_enc = pd.get_dummies(row_df).reindex(columns=feature_names, fill_value=0)
prob = model.predict_proba(scaler.transform(row_enc))[0][1]
label, badge_cls, emoji = get_risk(prob)
clv = customer['MonthlyCharges'] * customer['tenure'] * .25 * 83

m1, m2, m3, m4 = st.columns(4)
m1.metric("Churn Probability", f"{prob:.1%}")
m2.metric("Customer Lifetime Value", f"₹{clv:,.0f}")
m3.metric("Tenure", f"{int(customer['tenure'])} months")
m4.metric("Monthly Charges", f"₹{customer['MonthlyCharges']*83:.0f}")

gauge_col = '#f43f5e' if prob>.65 else '#f59e0b' if prob>.35 else '#22c55e'
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob*100,
    number={'suffix':'%','font':{'size':42,'color':'#e2e8f0'}},
    gauge=dict(
        axis=dict(range=[0,100], tickfont=dict(color='#4a5568')),
        bar=dict(color=gauge_col),
        bgcolor='#111827',
        steps=[dict(range=[0,35],color='rgba(34,197,94,.08)'),
               dict(range=[35,65],color='rgba(245,158,11,.08)'),
               dict(range=[65,100],color='rgba(244,63,94,.08)')]),
    title=dict(text=f"Churn Risk Meter  {emoji}", font=dict(color='#94a3b8',size=14))))
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280,
                  font=dict(family='Inter'), margin=dict(l=40,r=40,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 👤 Customer Profile")
profile_cols = ['gender','SeniorCitizen','Partner','Dependents','tenure','Contract',
                'InternetService','MonthlyCharges','TotalCharges','PaymentMethod']
existing = [c for c in profile_cols if c in customer.index]
half = len(existing)//2
p1, p2 = st.columns(2)
with p1:
    st.dataframe(pd.DataFrame({'Feature':existing[:half],'Value':[str(customer[c]) for c in existing[:half]]}).set_index('Feature'), use_container_width=True)
with p2:
    st.dataframe(pd.DataFrame({'Feature':existing[half:],'Value':[str(customer[c]) for c in existing[half:]]}).set_index('Feature'), use_container_width=True)

st.markdown("### 🎯 Recommended Retention Action")
s_name, s_reason, s_cost, s_lift = strategy(prob, clv)
st.markdown(f"""
<div class='card'>
  <div style='font-size:1.1rem;font-weight:700;color:#a78bfa;margin-bottom:8px;'>{s_name}</div>
  <div style='color:#94a3b8;margin-bottom:16px;'>{s_reason}</div>
  <div style='display:flex;gap:32px;flex-wrap:wrap;'>
    <div><div style='color:#4a5568;font-size:.75rem;'>Estimated Cost</div>
         <div style='color:#e2e8f0;font-weight:700;font-size:1.1rem;'>{s_cost}</div></div>
    <div><div style='color:#4a5568;font-size:.75rem;'>Expected Lift</div>
         <div style='color:#e2e8f0;font-weight:700;font-size:1.1rem;'>{s_lift}</div></div>
    <div><div style='color:#4a5568;font-size:.75rem;'>Risk Level</div>
         <span class='{badge_cls}'>{label}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)
