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

# ── SHAP (optional — gracefully skip if not installed) ──────────────────────
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

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

drop_cols = ['customerID','Churn','tenure_group','RFM_Segment','recency','frequency',
             'monetary','R_Score','F_Score','M_Score']
row = customer.drop([c for c in drop_cols if c in customer.index], errors='ignore')
row_df = pd.DataFrame([row])
row_enc = pd.get_dummies(row_df).reindex(columns=feature_names, fill_value=0)
row_scaled = scaler.transform(row_enc)
prob = model.predict_proba(row_scaled)[0][1]
label, badge_cls, emoji = get_risk(prob)
clv = customer['MonthlyCharges'] * customer['tenure'] * .25 * 83

# ── Top Metrics ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Churn Probability", f"{prob:.1%}")
m2.metric("Customer Lifetime Value", f"₹{clv:,.0f}")
m3.metric("Tenure", f"{int(customer['tenure'])} months")
m4.metric("Monthly Charges", f"₹{customer['MonthlyCharges']*83:.0f}")

# ── Gauge ───────────────────────────────────────────────────────────────────
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

# ── Customer Profile (FIXED) ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 👤 Customer Profile")

profile_cols = ['gender','SeniorCitizen','Partner','Dependents','tenure','Contract',
                'InternetService','MonthlyCharges','TotalCharges','PaymentMethod']
existing = [c for c in profile_cols if c in customer.index]

profile_data = []
for c in existing:
    val = customer[c]
    # format numbers nicely
    if isinstance(val, float):
        val = f"{val:.2f}"
    elif isinstance(val, (int, np.integer)):
        val = str(int(val))
    else:
        val = str(val)
    profile_data.append({"Feature": c, "Value": val})

profile_df = pd.DataFrame(profile_data)
half = len(profile_df) // 2

p1, p2 = st.columns(2)
with p1:
    st.markdown("**Account Details**")
    st.table(profile_df.iloc[:half].set_index("Feature"))
with p2:
    st.markdown("**Service Details**")
    st.table(profile_df.iloc[half:].set_index("Feature"))

# ── SHAP Explainability ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧠 Why is this customer at risk?")
st.caption("SHAP values show which features push the churn probability up (red) or down (blue).")

if not _HAS_SHAP:
    st.warning("SHAP not installed. Run: `pip install shap` then restart the app.")
else:
    try:
        with st.spinner("Calculating SHAP values..."):
            # Use TreeExplainer for tree models, fallback to LinearExplainer
            model_type = type(model).__name__
            if model_type in ['RandomForestClassifier','XGBClassifier',
                               'LGBMClassifier','DecisionTreeClassifier',
                               'VotingClassifier']:
                # For voting ensemble or tree models
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(row_enc)
                    # For binary classifiers shap_values returns list [class0, class1]
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1][0]
                    else:
                        sv = shap_vals[0]
                except Exception:
                    explainer = shap.Explainer(model, row_enc)
                    shap_vals = explainer(row_enc)
                    sv = shap_vals.values[0]
                    if sv.ndim > 1:
                        sv = sv[:, 1]
            else:
                explainer = shap.LinearExplainer(model, row_enc)
                shap_vals = explainer.shap_values(row_enc)
                sv = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[1][0]

        # Build top-N bar chart
        shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP': sv})
        shap_df['abs'] = shap_df['SHAP'].abs()
        shap_df = shap_df.sort_values('abs', ascending=False).head(12)
        shap_df = shap_df.sort_values('SHAP', ascending=True)

        colors = ['#f43f5e' if v > 0 else '#6366f1' for v in shap_df['SHAP']]

        fig_shap = go.Figure(go.Bar(
            x=shap_df['SHAP'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f}" for v in shap_df['SHAP']],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=11)
        ))
        fig_shap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(17,24,39,.85)',
            font=dict(color='#94a3b8', family='Inter'),
            margin=dict(l=20, r=80, t=30, b=20),
            height=420,
            xaxis=dict(
                title='SHAP value (impact on churn probability)',
                gridcolor='#1e2840',
                zerolinecolor='#4a5568',
                zerolinewidth=1.5
            ),
            yaxis=dict(gridcolor='#1e2840'),
            showlegend=False
        )

        # Add legend annotation
        fig_shap.add_annotation(
            x=shap_df['SHAP'].max() * 0.6, y=1,
            text="🔴 Increases churn risk",
            showarrow=False, font=dict(color='#f43f5e', size=11),
            xref='x', yref='paper'
        )
        fig_shap.add_annotation(
            x=shap_df['SHAP'].min() * 0.6, y=1,
            text="🔵 Decreases churn risk",
            showarrow=False, font=dict(color='#6366f1', size=11),
            xref='x', yref='paper'
        )

        st.plotly_chart(fig_shap, use_container_width=True)

        # Top 3 reasons in plain English
        top_risk = shap_df[shap_df['SHAP'] > 0].sort_values('SHAP', ascending=False).head(3)
        top_safe = shap_df[shap_df['SHAP'] < 0].sort_values('SHAP').head(2)

        if not top_risk.empty:
            st.markdown("**Top reasons driving churn risk:**")
            for _, r in top_risk.iterrows():
                feat_val = row_enc[r['Feature']].values[0] if r['Feature'] in row_enc.columns else 'N/A'
                st.markdown(f"- 🔴 **{r['Feature']}** = `{feat_val}` &nbsp; (impact: +{r['SHAP']:.3f})")

        if not top_safe.empty:
            st.markdown("**Factors reducing churn risk:**")
            for _, r in top_safe.iterrows():
                feat_val = row_enc[r['Feature']].values[0] if r['Feature'] in row_enc.columns else 'N/A'
                st.markdown(f"- 🔵 **{r['Feature']}** = `{feat_val}` &nbsp; (impact: {r['SHAP']:.3f})")

    except Exception as e:
        st.error(f"SHAP calculation failed: {e}")
        st.info("This can happen with Voting Ensemble models. Try re-running `python train_models.py`.")

# ── Recommended Retention Action ────────────────────────────────────────────
st.markdown("---")
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