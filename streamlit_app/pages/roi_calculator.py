"""ROI Calculator — standalone Streamlit page.
Business metrics derived from real Telco dataset, not hardcoded.
"""
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
              font=dict(color='#94a3b8', family='Inter'), margin=dict(l=20,r=20,t=44,b=20),
              title_font=dict(color='#e2e8f0', size=15))

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#e2e8f0;'>💰 Retention ROI Calculator</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;margin-top:-10px;'>Business metrics derived from your real Telco dataset — not assumptions</p>",
            unsafe_allow_html=True)

df_raw = load_main_data()
if df_raw is None:
    st.error("Data not found. Run `python train_models.py` first.")
    st.stop()

df = df_raw.copy()

# ── Derive Real Business Metrics from Dataset ────────────────────────────────
# These numbers come from the actual Telco CSV, not hardcoded values

# 1. Real churn rate per contract type → used to compute data-driven lift
churn_col = 'Churn'  # already encoded as 0/1 in processed data
if churn_col in df.columns:
    churn_by_contract = df.groupby('Contract')[churn_col].mean()
    base_churn         = df[churn_col].mean()           # overall churn rate

    # Lift = how much churn drops if customer moves to a better contract
    # Computed from actual churn rate differences in your dataset
    mtm_churn  = churn_by_contract.get('Month-to-month', 0.43)
    one_churn  = churn_by_contract.get('One year',       0.11)
    two_churn  = churn_by_contract.get('Two year',       0.03)

    # Retention lift = churn rate reduction achievable per strategy tier
    # Premium: moves customer behaviour toward "Two year" equivalent
    # Standard: moves toward "One year" equivalent
    # Basic/Minimal: partial improvement
    lift_premium  = round(float(mtm_churn - two_churn), 2)   # ~0.40 from real data
    lift_standard = round(float(mtm_churn - one_churn), 2)   # ~0.32 from real data
    lift_basic    = round(float(lift_standard * 0.6),   2)   # ~0.19
    lift_minimal  = round(float(lift_standard * 0.25),  2)   # ~0.08

    data_note = (f"Lift rates derived from your dataset: "
                 f"Month-to-month churn={mtm_churn:.1%}, "
                 f"One-year={one_churn:.1%}, "
                 f"Two-year={two_churn:.1%}")
else:
    # Fallback if Churn column missing
    lift_premium, lift_standard, lift_basic, lift_minimal = 0.40, 0.30, 0.18, 0.08
    base_churn = 0.265
    data_note  = "Using fallback lift rates (Churn column not found in processed data)"

# 2. Real avg customer lifetime from data
# E(lifetime) = 1 / churn_rate  (geometric distribution of survival)
data_avg_lifetime = round(1 / base_churn) if base_churn > 0 else 24

# 3. Real avg monthly charges from data
data_avg_monthly = float(df['MonthlyCharges'].mean())

# Strategy definitions — costs are business inputs (sliders), lifts are data-derived
STRATEGIES = {
    "🎯 Premium: Manager + 20% Discount": {
        "lift": lift_premium,
        "target": "High CLV at-risk",
        "desc": f"Dedicated manager + 20% discount. Lift={lift_premium:.0%} (derived from contract churn gap in your data)"
    },
    "💳 Standard: 15% Discount Campaign": {
        "lift": lift_standard,
        "target": "Medium CLV at-risk",
        "desc": f"Personalised 15% discount via email/SMS. Lift={lift_standard:.0%} (one-year vs month-to-month gap)"
    },
    "📱 Basic: Loyalty Rewards + Upgrade": {
        "lift": lift_basic,
        "target": "Low-medium risk",
        "desc": f"Free service upgrade 3 months. Lift={lift_basic:.0%} (60% of standard lift)"
    },
    "📧 Minimal: Email Re-engagement": {
        "lift": lift_minimal,
        "target": "Low CLV",
        "desc": f"Automated nurture sequence. Lift={lift_minimal:.0%} (25% of standard lift)"
    },
}

# ── Show Data-Derived Insights ────────────────────────────────────────────────
with st.expander("📊 How lift rates are calculated from your data", expanded=False):
    st.caption(data_note)
    if churn_col in df.columns:
        st.markdown("**Churn rate by contract type (from your Telco dataset):**")
        cbd = churn_by_contract.reset_index()
        cbd.columns = ['Contract Type', 'Churn Rate']
        cbd['Churn Rate'] = cbd['Churn Rate'].apply(lambda x: f"{x:.1%}")
        cbd['Retention Lift vs Month-to-month'] = [
            "Baseline", f"+{lift_standard:.0%}", f"+{lift_premium:.0%}"
        ][:len(cbd)]
        st.dataframe(cbd.set_index('Contract Type'), use_container_width=True)
    st.markdown(f"""
    - **Base churn rate:** `{base_churn:.1%}` (from your {len(df):,} customers)
    - **Data-implied avg lifetime:** `{data_avg_lifetime} months` (= 1 ÷ churn rate)
    - **Avg monthly charge:** `₹{data_avg_monthly*83:,.0f}` (converted at slider rate)
    - Campaign **costs** are your business inputs — adjust with sliders below
    """)

# ── Business Parameter Sliders ────────────────────────────────────────────────
st.markdown("### ⚙️ Business Parameters")
st.caption("Lift rates come from your data. Set your cost assumptions and exchange rate below.")

p1, p2, p3 = st.columns(3)
with p1:
    margin = st.slider("Gross Margin (%)", 10, 60, 25,
                       help="Telecom industry avg is 20-35%") / 100
    inr    = st.slider("USD → INR Rate", 75, 95, 83)
with p2:
    thresh = st.slider("Flag at-risk above churn prob (%)", 30, 80, 50,
                       help="Lower = cast wider net, higher = precision targeting") / 100
    avg_life = st.slider("Avg Customer Lifetime (months)", 12, 60, data_avg_lifetime,
                         help=f"Data suggests {data_avg_lifetime} months based on your churn rate")
with p3:
    strategy_name = st.selectbox("Retention Strategy", list(STRATEGIES.keys()))
    st.markdown("<br>", unsafe_allow_html=True)
    # Cost per customer — the ONE thing that needs business input
    cost_per_customer = st.number_input(
        "Campaign cost per customer (₹)",
        min_value=10, max_value=5000,
        value={"🎯 Premium: Manager + 20% Discount": 500,
               "💳 Standard: 15% Discount Campaign": 200,
               "📱 Basic: Loyalty Rewards + Upgrade": 100,
               "📧 Minimal: Email Re-engagement":      10}[strategy_name],
        step=10,
        help="Only input that can't come from data — set based on your campaign budget"
    )

strat = STRATEGIES[strategy_name]

# ── Get Real Churn Probabilities from Trained Model ───────────────────────────
if models_trained():
    mdl = load_model('best_model')
    sc  = load_scaler()
    fn  = load_feature_names()
    if mdl and sc and fn:
        drop = ['customerID', 'Churn', 'tenure_group', 'RFM_Segment',
                'recency', 'frequency', 'monetary', 'R_Score', 'F_Score', 'M_Score']
        Xr = df.drop([c for c in drop if c in df.columns], axis=1)
        Xe = pd.get_dummies(Xr).reindex(columns=fn, fill_value=0)
        df['churn_prob'] = mdl.predict_proba(sc.transform(Xe))[:, 1]
        prob_source = "ML model (best_model.pkl)"
    else:
        df['churn_prob'] = (1 - (df['tenure'] / df['tenure'].max())) * 0.6
        prob_source = "Fallback heuristic (model files missing)"
else:
    df['churn_prob'] = (1 - (df['tenure'] / df['tenure'].max())) * 0.6
    prob_source = "Fallback heuristic (models not trained)"

# ── Data-Driven CLV Formula ───────────────────────────────────────────────────
# CLV = MonthlyCharges × expected_lifetime × margin × INR_rate
# expected_lifetime here = avg_life slider (defaulted to data-derived value)
# This is the standard telecom CLV formula — not a magic number
df['expected_lifetime'] = avg_life  # months; data-derived default
df['CLV'] = df['MonthlyCharges'] * df['expected_lifetime'] * margin * inr

# ── Campaign Calculations ─────────────────────────────────────────────────────
at_risk      = df[df['churn_prob'] >= thresh].copy()
n            = len(at_risk)
campaign_cost = n * cost_per_customer
saved_n      = int(n * strat['lift'])
clv_saved    = at_risk.nlargest(saved_n, 'CLV')['CLV'].sum() if saved_n > 0 else 0
net          = clv_saved - campaign_cost
roi_pct      = (net / campaign_cost * 100) if campaign_cost > 0 else 0
roi_color    = '#22c55e' if roi_pct > 100 else '#f59e0b' if roi_pct > 0 else '#f43f5e'

# ── Campaign Impact Metrics ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("<h3 style='color:#e2e8f0;'>📊 Campaign Impact</h3>", unsafe_allow_html=True)
st.caption(f"Churn probabilities from: {prob_source}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("At-Risk Customers",   f"{n:,}",
          delta=f"{n/len(df):.1%} of base", delta_color="inverse")
k2.metric("Revenue at Risk",     f"₹{at_risk['CLV'].sum():,.0f}")
k3.metric("Campaign Cost",       f"₹{campaign_cost:,.0f}")
k4.metric("Est. Revenue Saved",  f"₹{clv_saved:,.0f}")
k5.metric("Net ROI",             f"{roi_pct:.0f}%",
          delta="vs. no action",
          delta_color="normal" if roi_pct > 0 else "inverse")

st.markdown(f"""
<div class='card' style='border-top:3px solid {roi_color};margin-top:16px;'>
  <div style='font-size:1.1rem;font-weight:700;color:#a78bfa;margin-bottom:6px;'>{strategy_name}</div>
  <div style='color:#94a3b8;font-size:.85rem;margin-bottom:14px;'>{strat["desc"]}</div>
  <div style='display:flex;gap:36px;flex-wrap:wrap;'>
    <div><div style='color:#4a5568;font-size:.75rem;'>Cost per customer</div>
         <div style='color:#e2e8f0;font-weight:700;'>₹{cost_per_customer:,}</div></div>
    <div><div style='color:#4a5568;font-size:.75rem;'>Retention lift (data-derived)</div>
         <div style='color:#e2e8f0;font-weight:700;'>{strat["lift"]:.0%}</div></div>
    <div><div style='color:#4a5568;font-size:.75rem;'>Customers saved</div>
         <div style='color:#e2e8f0;font-weight:700;'>{saved_n:,}</div></div>
    <div><div style='color:#4a5568;font-size:.75rem;'>Net ROI</div>
         <div style='color:{roi_color};font-weight:800;font-size:1.3rem;'>{roi_pct:.0f}%</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="large")
with c1:
    fig_b = go.Figure(go.Bar(
        x=['Campaign Cost', 'Revenue Saved', 'Net Gain'],
        y=[campaign_cost, clv_saved, net],
        marker_color=['#f43f5e', '#22c55e', roi_color],
        text=[f'₹{v:,.0f}' for v in [campaign_cost, clv_saved, net]],
        textposition='outside', textfont=dict(color='#e2e8f0')))
    fig_b.update_layout(**LAYOUT, title="Cost vs Revenue Saved", height=340,
                        yaxis=dict(gridcolor='#1e2840'),
                        xaxis=dict(gridcolor='#1e2840'))
    st.plotly_chart(fig_b, use_container_width=True)

with c2:
    fig_w = go.Figure(go.Waterfall(
        orientation='v',
        measure=['relative', 'relative', 'total'],
        x=['Revenue Saved', 'Campaign Cost', 'Net'],
        y=[clv_saved, -campaign_cost, 0],
        connector=dict(line=dict(color='#1e2840')),
        increasing=dict(marker=dict(color='#22c55e')),
        decreasing=dict(marker=dict(color='#f43f5e')),
        totals=dict(marker=dict(color=roi_color))))
    fig_w.update_layout(**LAYOUT, title="ROI Waterfall", height=340,
                        yaxis=dict(gridcolor='#1e2840'),
                        xaxis=dict(gridcolor='#1e2840'))
    st.plotly_chart(fig_w, use_container_width=True)

# ── All Strategies Compared ───────────────────────────────────────────────────
st.markdown("<h3 style='color:#e2e8f0;'>🔄 All Strategies Compared</h3>", unsafe_allow_html=True)
st.caption("Lift rates are data-derived for all strategies. Only cost differs.")

COST_DEFAULTS = {
    "🎯 Premium: Manager + 20% Discount": 500,
    "💳 Standard: 15% Discount Campaign": 200,
    "📱 Basic: Loyalty Rewards + Upgrade": 100,
    "📧 Minimal: Email Re-engagement":     10,
}

cmp = []
for sn, s in STRATEGIES.items():
    cc  = n * COST_DEFAULTS[sn]
    sv  = at_risk.nlargest(int(n * s['lift']), 'CLV')['CLV'].sum() if int(n * s['lift']) > 0 else 0
    net_s = sv - cc
    roi_s = round((net_s / cc * 100), 1) if cc > 0 else 0
    cmp.append({
        'Strategy':          sn.split(":")[0].strip(),
        'Lift (data-derived)': f"{s['lift']:.0%}",
        'Cost/Customer (₹)': COST_DEFAULTS[sn],
        'Total Cost (₹)':    int(cc),
        'Revenue Saved (₹)': int(sv),
        'Net (₹)':           int(net_s),
        'ROI (%)':           roi_s,
        'Customers Saved':   int(n * s['lift']),
    })

cmp_df = pd.DataFrame(cmp)
fig_cmp_tbl = go.Figure(go.Table(
    header=dict(
        values=list(cmp_df.columns),
        fill_color='#1e2840',
        font=dict(color='#e2e8f0', size=13, family='Inter'),
        align='left',
        height=36,
        line=dict(color='#2d3748', width=1)
    ),
    cells=dict(
        values=[cmp_df[c].tolist() for c in cmp_df.columns],
        fill_color=[
            ['rgba(30,40,64,0.9)'] * len(cmp_df),
            ['rgba(30,40,64,0.9)'] * len(cmp_df),
            ['rgba(30,40,64,0.9)'] * len(cmp_df),
            ['rgba(30,40,64,0.9)'] * len(cmp_df),
            ['rgba(30,40,64,0.9)'] * len(cmp_df),
            ['rgba(30,40,64,0.9)'] * len(cmp_df),
            ['rgba(34,197,94,0.15)' if v > 100
             else 'rgba(245,158,11,0.15)' if v > 0
             else 'rgba(244,63,94,0.15)' for v in cmp_df['ROI (%)']],
            ['rgba(30,40,64,0.9)'] * len(cmp_df),
        ],
        font=dict(color='#cbd5e1', size=12, family='Inter'),
        align='left',
        height=34,
        line=dict(color='#2d3748', width=1)
    )
))
fig_cmp_tbl.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=0, b=0),
    height=46 + 36 + 34 * len(cmp_df)
)
st.plotly_chart(fig_cmp_tbl, use_container_width=True)

fig_cmp = go.Figure(go.Bar(
    x=cmp_df['Strategy'],
    y=cmp_df['ROI (%)'],
    marker_color=['#22c55e' if v > 100 else '#f59e0b' if v > 0 else '#f43f5e'
                  for v in cmp_df['ROI (%)']],
    text=cmp_df['ROI (%)'].apply(lambda x: f"{x:.0f}%"),
    textposition='outside',
    textfont=dict(color='#e2e8f0')))
fig_cmp.update_layout(**LAYOUT, title="ROI by Strategy (data-derived lifts)", height=320,
                      yaxis=dict(title='ROI (%)', gridcolor='#1e2840'),
                      xaxis=dict(gridcolor='#1e2840'),
                      showlegend=False)
st.plotly_chart(fig_cmp, use_container_width=True)

# ── CLV Distribution of At-Risk Customers ────────────────────────────────────
st.markdown("<h3 style='color:#e2e8f0;'>💎 CLV Distribution — At-Risk Customers</h3>",
            unsafe_allow_html=True)
st.caption("Prioritise retention spend on the right tail — highest CLV churners.")

fig_clv = go.Figure()
fig_clv.add_trace(go.Histogram(
    x=at_risk['CLV'], nbinsx=40,
    marker_color='#6366f1', opacity=0.8,
    name='At-risk CLV'))
fig_clv.add_vline(
    x=at_risk['CLV'].mean(),
    line=dict(color='#f59e0b', dash='dash', width=1.5),
    annotation_text=f"Mean ₹{at_risk['CLV'].mean():,.0f}",
    annotation_font_color='#f59e0b')
fig_clv.add_vline(
    x=at_risk['CLV'].quantile(0.75),
    line=dict(color='#f43f5e', dash='dot', width=1.5),
    annotation_text=f"75th pct ₹{at_risk['CLV'].quantile(0.75):,.0f}",
    annotation_font_color='#f43f5e')
fig_clv.update_layout(**LAYOUT, height=320,
                      title=f"CLV distribution across {n:,} at-risk customers",
                      xaxis=dict(title='Customer Lifetime Value (₹)', gridcolor='#1e2840'),
                      yaxis=dict(title='Count', gridcolor='#1e2840'))
st.plotly_chart(fig_clv, use_container_width=True)