"""
ChurnShield — Home / Landing Page
Run: streamlit run streamlit_app/app.py  (from project root)
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
_sapp = os.path.join(ROOT, 'streamlit_app')
if _sapp not in sys.path: sys.path.insert(0, _sapp)

import streamlit as st
st.set_page_config(
    page_title="ChurnShield | Customer Intelligence Platform",
    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from utils.styles import apply_styles, sidebar_brand
apply_styles(); sidebar_brand()

from utils.data_loader import load_main_data, models_trained, data_available

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2.5rem 0 1rem;'>
  <div style='font-size:4rem; margin-bottom:.5rem;'>🛡️</div>
  <h1 style='font-size:3rem; margin-bottom:.5rem;'>ChurnShield</h1>
  <p style='font-size:1.2rem; color:#94a3b8; max-width:600px; margin:0 auto 1.5rem;'>
    Customer Intelligence Platform — Predict, Segment, and Retain your subscribers
  </p>
</div>
""", unsafe_allow_html=True)

# ── Status banner ────────────────────────────────────────────────────────────
if data_available():
    if models_trained():
        st.success("✅  Dataset loaded & models trained — all pages are live!", icon="🚀")
    else:
        st.warning("⚠️  Dataset found but models not trained. Run `python train_models.py`.", icon="⚙️")
        st.code("python train_models.py")
else:
    st.error("⚠️  No dataset. Run the setup commands below.", icon="🚨")
    st.code("python generate_sample_data.py\npython train_models.py")

st.markdown("---")

# ── Feature cards ────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Navigate the Platform")
r1 = st.columns(3, gap="large")
cards = [
    ("🏠", "Overview", "Churn KPIs, distribution charts, payment method analysis, and heatmaps across all customers.", "#6366f1"),
    ("🔍", "Customer Lookup", "Search any customer ID → get a live churn probability gauge + personalised retention recommendation.", "#f43f5e"),
    ("👥", "Segments", "RFM lifecycle segments + interactive K-Means clustering with Elbow/Silhouette curve.", "#22c55e"),
    ("📈", "Model Performance", "ROC & PR curves for all 5 models, confusion matrix, feature importance, and threshold optimizer.", "#f59e0b"),
    ("🎯", "What-If Simulator", "Change contract type, tenure, and services in real time — watch churn probability update live.", "#06b6d4"),
    ("💰", "ROI Calculator", "Simulate retention campaigns. Get ₹ cost, revenue saved, and ROI for 4 strategies.", "#a78bfa"),
]
for i, (icon, title, desc, color) in enumerate(cards):
    with r1[i % 3]:
        st.markdown(f"""
<div class='card' style='border-top:3px solid {color}; min-height:180px;'>
  <div style='font-size:2rem; margin-bottom:.5rem;'>{icon}</div>
  <div style='font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:.5rem;'>{title}</div>
  <div style='font-size:.85rem; color:#94a3b8; line-height:1.5;'>{desc}</div>
</div>""", unsafe_allow_html=True)
    if i == 2:
        st.markdown("<br>", unsafe_allow_html=True)
        r1 = st.columns(3, gap="large")

st.markdown("---")

# ── Quick stats if data available ────────────────────────────────────────────
if data_available():
    df = load_main_data()
    st.markdown("### ⚡ Quick Stats")
    q1,q2,q3,q4,q5 = st.columns(5)
    q1.metric("Total Customers", f"{len(df):,}")
    q2.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
    q3.metric("Avg Tenure", f"{df['tenure'].mean():.1f} mo")
    q4.metric("Avg Monthly (₹)", f"₹{df['MonthlyCharges'].mean()*83:.0f}")
    q5.metric("Features", "21 + Engineered")

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#374151; font-size:.8rem; padding:1rem 0;'>
  Built for ML Coursework · Telecom Churn Prediction · India Market Focus
</div>""", unsafe_allow_html=True)
