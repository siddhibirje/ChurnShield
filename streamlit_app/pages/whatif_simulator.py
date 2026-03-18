"""What-If Simulator — standalone Streamlit page."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
_sapp = os.path.join(ROOT, 'streamlit_app')
if _sapp not in sys.path: sys.path.insert(0, _sapp)

import streamlit as st
st.set_page_config(page_title="What-If Simulator | ChurnShield", page_icon="🎯", layout="wide")

from utils.styles import apply_styles, sidebar_brand
apply_styles(); sidebar_brand()

import pandas as pd, numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_model, load_scaler, load_feature_names, models_trained

st.markdown("# 🎯 What-If Simulator")
st.markdown("<p style='color:#94a3b8;margin-top:-10px;'>Adjust customer attributes and watch churn probability update live</p>",
            unsafe_allow_html=True)

if not models_trained():
    st.warning("⚠️ Models not trained yet."); st.stop()

model = load_model('best_model')
scaler = load_scaler()
feature_names = load_feature_names()
if not all([model, scaler, feature_names]):
    st.error("Model files missing. Run `python train_models.py`."); st.stop()

st.markdown("### ⚙️ Configure Customer Profile")
st.markdown("---")
ca, cb, cc = st.columns(3)

with ca:
    st.markdown("**📋 Subscription**")
    contract = st.selectbox("Contract", ['Month-to-month','One year','Two year'])
    internet = st.selectbox("Internet Service", ['Fiber optic','DSL','No'])
    tenure   = st.slider("Tenure (months)", 0, 72, 12)
    monthly  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, .5)

with cb:
    st.markdown("**🔧 Add-ons**")
    tech_sup  = st.selectbox("Tech Support",     ['No','Yes','No internet service'])
    online_sec= st.selectbox("Online Security",  ['No','Yes','No internet service'])
    o_backup  = st.selectbox("Online Backup",    ['No','Yes','No internet service'])
    streaming = st.selectbox("Streaming TV",     ['No','Yes','No internet service'])
    movies    = st.selectbox("Streaming Movies", ['No','Yes','No internet service'])

with cc:
    st.markdown("**👤 Demographics**")
    senior    = st.selectbox("Senior Citizen", [0,1], format_func=lambda x:"Yes" if x else "No")
    partner   = st.selectbox("Partner",        ['Yes','No'])
    dependents= st.selectbox("Dependents",     ['Yes','No'])
    payment   = st.selectbox("Payment Method", ['Electronic check','Mailed check',
                                                 'Bank transfer (automatic)','Credit card (automatic)'])
    paperless = st.selectbox("Paperless Billing", ['Yes','No'])
    phone     = st.selectbox("Phone Service", ['Yes','No'])
    mlines    = st.selectbox("Multiple Lines", ['No','Yes','No phone service'])

total_charges = monthly * tenure
engagement = sum([phone=='Yes', mlines=='Yes', online_sec=='Yes', o_backup=='Yes',
                  tech_sup=='Yes', streaming=='Yes', movies=='Yes'])
cpm = monthly if tenure==0 else total_charges/tenure

params = {
    'SeniorCitizen':senior,'tenure':tenure,'MonthlyCharges':monthly,
    'TotalCharges':total_charges,'engagement_score':engagement,'charges_per_month':cpm,
    'Partner':partner,'Dependents':dependents,'PhoneService':phone,'MultipleLines':mlines,
    'InternetService':internet,'OnlineSecurity':online_sec,'OnlineBackup':o_backup,
    'DeviceProtection':'No','TechSupport':tech_sup,'StreamingTV':streaming,
    'StreamingMovies':movies,'Contract':contract,'PaperlessBilling':paperless,
    'PaymentMethod':payment,'gender':'Male',
}
row_enc = pd.get_dummies(pd.DataFrame([params])).reindex(columns=feature_names, fill_value=0)
prob = model.predict_proba(scaler.transform(row_enc))[0][1]

st.markdown("---")
st.markdown("## 📊 Live Prediction")
gauge_col = '#f43f5e' if prob>.65 else '#f59e0b' if prob>.35 else '#22c55e'
risk_label = "🔴 HIGH RISK" if prob>.65 else "🟡 MEDIUM RISK" if prob>.35 else "🟢 LOW RISK"

r1,r2,r3,r4 = st.columns(4)
r1.metric("Churn Probability", f"{prob:.1%}")
r2.metric("Engagement Score", f"{engagement}/7")
r3.metric("Est. CLV (₹)", f"₹{monthly*max(tenure,1)*.25*83:,.0f}")
r4.metric("Risk Level", risk_label)

fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=prob*100,
    delta={'reference':50,'valueformat':'.1f',
           'increasing':{'color':'#f43f5e'},'decreasing':{'color':'#22c55e'}},
    number={'suffix':'%','font':{'size':44,'color':'#e2e8f0'}},
    gauge=dict(
        axis=dict(range=[0,100], tickfont=dict(color='#4a5568')),
        bar=dict(color=gauge_col), bgcolor='#111827',
        steps=[dict(range=[0,35],color='rgba(34,197,94,.08)'),
               dict(range=[35,65],color='rgba(245,158,11,.08)'),
               dict(range=[65,100],color='rgba(244,63,94,.08)')]),
    title=dict(text="Churn Risk Meter", font=dict(color='#94a3b8'))))
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300,
                  font=dict(family='Inter'), margin=dict(l=40,r=40,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 🔬 Key Risk Factors Detected")
factors = []
if contract == 'Month-to-month': factors.append(("📋 Month-to-Month Contract","Biggest churn driver — no long-term lock-in","neg"))
if internet == 'Fiber optic' and monthly > 80: factors.append(("🔥 High-Cost Fiber","Price-sensitive segment churns fastest","neg"))
if tenure < 12: factors.append(("⏱️ New Customer (<12 mo)","Early-life churn period — high risk window","neg"))
if tech_sup == 'No' and internet != 'No': factors.append(("🛠️ No Tech Support","Service frustration increases churn probability","neg"))
if payment == 'Electronic check': factors.append(("💳 Electronic Check","Correlated with higher churn in dataset","neg"))
if engagement >= 5: factors.append(("✅ High Engagement (5+ services)","Multiple services reduce churn significantly","pos"))
if tenure >= 36: factors.append(("✅ Long-Term Customer","Strongest single retention signal","pos"))
if contract == 'Two year': factors.append(("✅ Two-Year Contract","Lowest churn rate of any contract type","pos"))

if not factors:
    st.info("No strong risk factors detected for this profile.")
for factor, desc, ftype in factors:
    color = '#f43f5e' if ftype=="neg" else '#22c55e'
    icon = '⬆️' if ftype=="neg" else '⬇️'
    st.markdown(f"""
<div style='display:flex;align-items:center;gap:12px;padding:10px 16px;margin-bottom:6px;
            border-radius:10px;background:rgba(17,24,39,.9);border-left:3px solid {color};'>
  <span style='font-size:.85rem;font-weight:600;color:{color};'>{icon} {factor}</span>
  <span style='font-size:.8rem;color:#94a3b8;'>— {desc}</span>
</div>""", unsafe_allow_html=True)
