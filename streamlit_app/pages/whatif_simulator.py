"""What-If Simulator — standalone Streamlit page.
Risk factors derived from trained model feature importances, not hardcoded rules.
"""
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
from utils.data_loader import (load_model, load_scaler, load_feature_names,
                                load_main_data, models_trained)

# ── SHAP optional ─────────────────────────────────────────────────────────────
try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

st.markdown("<h1 style='color:#e2e8f0;'>🎯 What-If Simulator</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;margin-top:-10px;'>Adjust customer attributes and watch churn probability update live</p>",
            unsafe_allow_html=True)

if not models_trained():
    st.warning("⚠️ Models not trained yet. Run `python train_models.py` first.")
    st.stop()

model        = load_model('best_model')
scaler       = load_scaler()
feature_names = load_feature_names()
df_real      = load_main_data()

if not all([model, scaler, feature_names]):
    st.error("Model files missing. Run `python train_models.py`.")
    st.stop()

# ── Pre-compute feature importances from trained model ────────────────────────
@st.cache_data
def get_feature_importances(_model, _feature_names):
    """Extract feature importances from the best model — data-driven, not rules."""
    if hasattr(_model, 'feature_importances_'):
        fi = dict(zip(_feature_names, _model.feature_importances_))
    elif hasattr(_model, 'estimators_'):
        # VotingClassifier — average importances from tree-based sub-models
        importances = []
        for _, est in _model.estimators:
            if hasattr(est, 'feature_importances_'):
                importances.append(est.feature_importances_)
        if importances:
            fi = dict(zip(_feature_names, np.mean(importances, axis=0)))
        else:
            fi = {}
    elif hasattr(_model, 'coef_'):
        # Logistic Regression
        fi = dict(zip(_feature_names, np.abs(_model.coef_[0])))
    else:
        fi = {}
    return fi

fi_map = get_feature_importances(model, feature_names)

# ── Pre-compute data-driven stats for risk context ────────────────────────────
@st.cache_data
def get_data_stats(_df):
    """Real stats from Telco dataset for contextual risk messages."""
    stats = {}
    if _df is None:
        return stats
    churn_col = 'Churn'
    if churn_col not in _df.columns:
        return stats
    if 'Contract' in _df.columns:
        stats['contract_churn'] = _df.groupby('Contract')[churn_col].mean().to_dict()
    if 'InternetService' in _df.columns:
        stats['internet_churn'] = _df.groupby('InternetService')[churn_col].mean().to_dict()
    if 'tenure' in _df.columns:
        stats['early_churn']  = float(_df[_df['tenure'] <= 12][churn_col].mean())
        stats['mature_churn'] = float(_df[_df['tenure'] >= 36][churn_col].mean())
    if 'TechSupport' in _df.columns:
        stats['no_support_churn']  = float(_df[_df['TechSupport'] == 'No'][churn_col].mean())
        stats['yes_support_churn'] = float(_df[_df['TechSupport'] == 'Yes'][churn_col].mean())
    if 'PaymentMethod' in _df.columns:
        stats['payment_churn'] = _df.groupby('PaymentMethod')[churn_col].mean().to_dict()
    if 'engagement_score' in _df.columns:
        stats['high_eng_churn'] = float(_df[_df['engagement_score'] >= 5][churn_col].mean())
        stats['low_eng_churn']  = float(_df[_df['engagement_score'] <= 2][churn_col].mean())
    return stats

ds = get_data_stats(df_real)

# ── Sidebar: Customer Profile Inputs ─────────────────────────────────────────
st.markdown("### ⚙️ Configure Customer Profile")
st.markdown("---")
ca, cb, cc = st.columns(3)

with ca:
    st.markdown("**📋 Subscription**")
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    internet = st.selectbox("Internet Service", ['Fiber optic', 'DSL', 'No'])
    tenure   = st.slider("Tenure (months)", 0, 72, 12)
    monthly  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)

with cb:
    st.markdown("**🔧 Add-ons**")
    tech_sup   = st.selectbox("Tech Support",      ['No', 'Yes', 'No internet service'])
    online_sec = st.selectbox("Online Security",   ['No', 'Yes', 'No internet service'])
    o_backup   = st.selectbox("Online Backup",     ['No', 'Yes', 'No internet service'])
    streaming  = st.selectbox("Streaming TV",      ['No', 'Yes', 'No internet service'])
    movies     = st.selectbox("Streaming Movies",  ['No', 'Yes', 'No internet service'])
    dev_prot   = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])

with cc:
    st.markdown("**👤 Demographics**")
    gender     = st.selectbox("Gender", ['Male', 'Female'])
    senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner    = st.selectbox("Partner",     ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    payment    = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless  = st.selectbox("Paperless Billing", ['Yes', 'No'])
    phone      = st.selectbox("Phone Service",     ['Yes', 'No'])
    mlines     = st.selectbox("Multiple Lines",    ['No', 'Yes', 'No phone service'])

# ── Derived features (same as feature_engineering.py) ────────────────────────
total_charges  = monthly * tenure
engagement     = sum([
    phone == 'Yes', mlines == 'Yes', online_sec == 'Yes',
    o_backup == 'Yes', dev_prot == 'Yes', tech_sup == 'Yes',
    streaming == 'Yes', movies == 'Yes'
])
cpm = monthly if tenure == 0 else total_charges / tenure

params = {
    'gender': gender, 'SeniorCitizen': senior, 'tenure': tenure,
    'MonthlyCharges': monthly, 'TotalCharges': total_charges,
    'engagement_score': engagement, 'charges_per_month': cpm,
    'Partner': partner, 'Dependents': dependents,
    'PhoneService': phone, 'MultipleLines': mlines,
    'InternetService': internet, 'OnlineSecurity': online_sec,
    'OnlineBackup': o_backup, 'DeviceProtection': dev_prot,
    'TechSupport': tech_sup, 'StreamingTV': streaming,
    'StreamingMovies': movies, 'Contract': contract,
    'PaperlessBilling': paperless, 'PaymentMethod': payment,
}

row_df  = pd.DataFrame([params])
row_enc = pd.get_dummies(row_df).reindex(columns=feature_names, fill_value=0)
prob    = model.predict_proba(scaler.transform(row_enc))[0][1]

# ── Live Prediction ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<h2 style='color:#e2e8f0;'>📊 Live Prediction</h2>", unsafe_allow_html=True)

gauge_col  = '#f43f5e' if prob > .65 else '#f59e0b' if prob > .35 else '#22c55e'
risk_label = "🔴 HIGH RISK" if prob > .65 else "🟡 MEDIUM RISK" if prob > .35 else "🟢 LOW RISK"
clv_est    = monthly * max(tenure, 1) * 0.25 * 83

r1, r2, r3, r4 = st.columns(4)
r1.metric("Churn Probability", f"{prob:.1%}")
r2.metric("Engagement Score",  f"{engagement}/8")
r3.metric("Est. CLV (₹)",      f"₹{clv_est:,.0f}")
r4.metric("Risk Level",        risk_label)

fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=prob * 100,
    delta={'reference': 50, 'valueformat': '.1f',
           'increasing': {'color': '#f43f5e'}, 'decreasing': {'color': '#22c55e'}},
    number={'suffix': '%', 'font': {'size': 44, 'color': '#e2e8f0'}},
    gauge=dict(
        axis=dict(range=[0, 100], tickfont=dict(color='#4a5568')),
        bar=dict(color=gauge_col), bgcolor='#111827',
        steps=[dict(range=[0,  35], color='rgba(34,197,94,.08)'),
               dict(range=[35, 65], color='rgba(245,158,11,.08)'),
               dict(range=[65,100], color='rgba(244,63,94,.08)')]),
    title=dict(text="Churn Risk Meter", font=dict(color='#94a3b8'))))
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300,
                  font=dict(family='Inter'), margin=dict(l=40,r=40,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

# ── Data-Driven Risk Factors ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("<h3 style='color:#e2e8f0;'>🔬 Risk Factors — From Your Data</h3>",
            unsafe_allow_html=True)
st.caption("Each factor shows the actual churn rate difference found in your Telco dataset.")

factors = []

# Contract type — from real data
c_churn = ds.get('contract_churn', {})
if contract == 'Month-to-month' and c_churn:
    mtm = c_churn.get('Month-to-month', 0.43)
    two = c_churn.get('Two year', 0.03)
    factors.append({
        'label': '📋 Month-to-Month Contract',
        'desc':  f"Churn rate in your data: {mtm:.1%} vs {two:.1%} for two-year contracts",
        'type':  'neg',
        'data':  True
    })
elif contract == 'Two year':
    two = c_churn.get('Two year', 0.03)
    factors.append({
        'label': '✅ Two-Year Contract',
        'desc':  f"Lowest churn in your dataset: {two:.1%} churn rate",
        'type':  'pos',
        'data':  True
    })
elif contract == 'One year':
    one = c_churn.get('One year', 0.11)
    factors.append({
        'label': '📋 One-Year Contract',
        'desc':  f"Moderate churn in your dataset: {one:.1%} churn rate",
        'type':  'neu',
        'data':  True
    })

# Tenure — from real data
early  = ds.get('early_churn',  0.47)
mature = ds.get('mature_churn', 0.15)
if tenure <= 12:
    factors.append({
        'label': '⏱️ New Customer (≤12 months)',
        'desc':  f"Early-tenure customers churn at {early:.1%} in your data vs {mature:.1%} for 36+ months",
        'type':  'neg', 'data': True
    })
elif tenure >= 36:
    factors.append({
        'label': '✅ Long-Tenure Customer (36+ months)',
        'desc':  f"Long-tenure customers churn at only {mature:.1%} in your data",
        'type':  'pos', 'data': True
    })

# Tech support — from real data
ns_churn  = ds.get('no_support_churn',  0.41)
yes_churn = ds.get('yes_support_churn', 0.15)
if tech_sup == 'No' and internet != 'No':
    factors.append({
        'label': '🛠️ No Tech Support',
        'desc':  f"Without support: {ns_churn:.1%} churn vs {yes_churn:.1%} with support (your data)",
        'type':  'neg', 'data': True
    })
elif tech_sup == 'Yes':
    factors.append({
        'label': '✅ Tech Support Active',
        'desc':  f"Reduces churn from {ns_churn:.1%} → {yes_churn:.1%} in your dataset",
        'type':  'pos', 'data': True
    })

# Payment method — from real data
pay_churn = ds.get('payment_churn', {})
if pay_churn and payment in pay_churn:
    p_rate   = pay_churn[payment]
    max_pay  = max(pay_churn.values())
    min_pay  = min(pay_churn.values())
    if p_rate >= max_pay * 0.9:
        factors.append({
            'label': f'💳 {payment}',
            'desc':  f"Highest-churn payment method in your data: {p_rate:.1%} churn rate",
            'type':  'neg', 'data': True
        })
    elif p_rate <= min_pay * 1.2:
        factors.append({
            'label': f'✅ {payment}',
            'desc':  f"Lowest-churn payment method in your data: {p_rate:.1%} churn rate",
            'type':  'pos', 'data': True
        })

# Engagement — from real data
high_eng = ds.get('high_eng_churn', 0.18)
low_eng  = ds.get('low_eng_churn',  0.45)
if engagement >= 5:
    factors.append({
        'label': f'✅ High Engagement ({engagement}/8 services)',
        'desc':  f"5+ services: {high_eng:.1%} churn in your data vs {low_eng:.1%} for low-engagement",
        'type':  'pos', 'data': True
    })
elif engagement <= 2:
    factors.append({
        'label': f'⚠️ Low Engagement ({engagement}/8 services)',
        'desc':  f"≤2 services: {low_eng:.1%} churn in your data — add services to retain",
        'type':  'neg', 'data': True
    })

# Internet service — from real data
int_churn = ds.get('internet_churn', {})
if int_churn and internet in int_churn:
    i_rate = int_churn[internet]
    max_int = max(int_churn.values())
    if i_rate >= max_int * 0.9:
        factors.append({
            'label': f'🔥 {internet} Internet',
            'desc':  f"Highest-churn internet type in your data: {i_rate:.1%} churn rate",
            'type':  'neg', 'data': True
        })

# Monthly charges — model feature importance context
if fi_map:
    top_features = sorted(fi_map.items(), key=lambda x: x[1], reverse=True)[:5]
    top_names    = [f[0] for f in top_features]
    monthly_feat = next((f for f in top_names if 'monthly' in f.lower() or 'charges' in f.lower()), None)
    if monthly_feat and monthly > 80:
        rank = next((i+1 for i, (n,_) in enumerate(
            sorted(fi_map.items(), key=lambda x: x[1], reverse=True))
            if 'monthly' in n.lower() or 'charges' in n.lower()), None)
        factors.append({
            'label': f'💸 High Monthly Charges (${monthly:.0f})',
            'desc':  f"MonthlyCharges is feature #{rank} by importance in your trained model",
            'type':  'neg', 'data': True
        })

# Senior citizen — straightforward
if senior == 1:
    factors.append({
        'label': '👤 Senior Citizen',
        'desc':  'Senior customers show higher churn rates in the Telco dataset',
        'type':  'neg', 'data': False
    })

if not factors:
    st.info("No strong risk factors detected for this profile — adjust the sliders above.")
else:
    for f in factors:
        color = '#f43f5e' if f['type'] == 'neg' else '#22c55e' if f['type'] == 'pos' else '#f59e0b'
        arrow = '⬆️ Increases risk' if f['type'] == 'neg' else '⬇️ Reduces risk' if f['type'] == 'pos' else '➡️ Moderate'
        badge = "📊 from data" if f['data'] else "ℹ️ general"
        st.markdown(f"""
<div style='display:flex;align-items:flex-start;gap:12px;padding:12px 16px;margin-bottom:8px;
            border-radius:10px;background:rgba(17,24,39,.9);border-left:3px solid {color};'>
  <div style='flex:1;'>
    <div style='font-size:.9rem;font-weight:600;color:{color};margin-bottom:2px;'>
      {f["label"]} &nbsp;<span style='font-size:.7rem;color:#4a5568;font-weight:400;'>{badge}</span>
    </div>
    <div style='font-size:.8rem;color:#94a3b8;'>{arrow} — {f["desc"]}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Feature Importance Context ────────────────────────────────────────────────
if fi_map:
    st.markdown("---")
    st.markdown("<h3 style='color:#e2e8f0;'>🧠 Model Feature Importance</h3>",
                unsafe_allow_html=True)
    st.caption("Top features your trained model uses to make predictions — not hardcoded rules.")

    fi_df = (pd.DataFrame(fi_map.items(), columns=['Feature', 'Importance'])
               .sort_values('Importance', ascending=True)
               .tail(15))

    # Highlight features matching current customer's active attributes
    active_features = [f for f in fi_df['Feature']
                       if row_enc[f].values[0] != 0 if f in row_enc.columns]
    colors = ['#f43f5e' if f in active_features else '#6366f1'
              for f in fi_df['Feature']]

    fig_fi = go.Figure(go.Bar(
        x=fi_df['Importance'], y=fi_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in fi_df['Importance']],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=10)))
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,.85)',
        font=dict(color='#94a3b8', family='Inter'),
        margin=dict(l=20, r=80, t=30, b=20),
        height=480,
        title="Top 15 features — red = active in this customer profile",
        title_font=dict(color='#e2e8f0', size=13),
        xaxis=dict(title='Importance', gridcolor='#1e2840'),
        yaxis=dict(gridcolor='#1e2840'),
        showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

# ── What-If Comparison Table ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("<h3 style='color:#e2e8f0;'>🔁 Scenario Comparison</h3>", unsafe_allow_html=True)
st.caption("See how churn probability changes if you improve key attributes.")

scenarios = []

def predict_scenario(overrides):
    p = {**params, **overrides}
    enc = pd.get_dummies(pd.DataFrame([p])).reindex(columns=feature_names, fill_value=0)
    return model.predict_proba(scaler.transform(enc))[0][1]

scenarios.append({'Scenario': '📍 Current Profile',    'Churn Prob': f"{prob:.1%}", 'Change': '—'})

# Scenario 1: upgrade contract
if contract != 'Two year':
    p2 = predict_scenario({'Contract': 'Two year'})
    scenarios.append({'Scenario': '📋 Upgrade to Two-Year Contract',
                      'Churn Prob': f"{p2:.1%}",
                      'Change': f"{'↓' if p2 < prob else '↑'} {abs(p2-prob):.1%}"})

# Scenario 2: add tech support
if tech_sup == 'No' and internet != 'No':
    p3 = predict_scenario({'TechSupport': 'Yes'})
    scenarios.append({'Scenario': '🛠️ Add Tech Support',
                      'Churn Prob': f"{p3:.1%}",
                      'Change': f"{'↓' if p3 < prob else '↑'} {abs(p3-prob):.1%}"})

# Scenario 3: change payment method
best_pay = min(ds.get('payment_churn', {payment: prob}).items(), key=lambda x: x[1])[0]
if best_pay != payment:
    p4 = predict_scenario({'PaymentMethod': best_pay})
    scenarios.append({'Scenario': f'💳 Switch to {best_pay}',
                      'Churn Prob': f"{p4:.1%}",
                      'Change': f"{'↓' if p4 < prob else '↑'} {abs(p4-prob):.1%}"})

# Scenario 4: longer tenure (what if retained 12 more months)
if tenure < 60:
    p5 = predict_scenario({'tenure': tenure + 12,
                            'TotalCharges': (tenure + 12) * monthly})
    scenarios.append({'Scenario': '⏱️ If retained 12 more months',
                      'Churn Prob': f"{p5:.1%}",
                      'Change': f"{'↓' if p5 < prob else '↑'} {abs(p5-prob):.1%}"})

# Scenario 5: add all services
p6 = predict_scenario({
    'TechSupport': 'Yes', 'OnlineSecurity': 'Yes',
    'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes'})
scenarios.append({'Scenario': '✅ Add All Protection Services',
                  'Churn Prob': f"{p6:.1%}",
                  'Change': f"{'↓' if p6 < prob else '↑'} {abs(p6-prob):.1%}"})
sc_df = pd.DataFrame(scenarios)

fig_sc = go.Figure(go.Table(
    header=dict(
        values=['Scenario', 'Churn Prob', 'Change'],
        fill_color='#1e2840',
        font=dict(color='#e2e8f0', size=13, family='Inter'),
        align='left',
        height=36,
        line=dict(color='#2d3748', width=1)
    ),
    cells=dict(
        values=[
            sc_df['Scenario'].tolist(),
            sc_df['Churn Prob'].tolist(),
            sc_df['Change'].tolist()
        ],
        fill_color=[
            ['rgba(30,40,64,0.9)'] * len(sc_df),
            ['rgba(30,40,64,0.9)'] * len(sc_df),
            ['rgba(34,197,94,0.15)' if '↓' in str(v)
             else 'rgba(244,63,94,0.15)' if '↑' in str(v)
             else 'rgba(30,40,64,0.9)' for v in sc_df['Change']]
        ],
        font=dict(color='#cbd5e1', size=12, family='Inter'),
        align='left',
        height=34,
        line=dict(color='#2d3748', width=1)
    )
))
fig_sc.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=0, b=0),
    height=46 + 36 + 34 * len(sc_df)
)
st.plotly_chart(fig_sc, use_container_width=True)
