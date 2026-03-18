"""Shared CSS — import this in every page to apply the dark theme."""
import streamlit as st


def apply_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Dark background */
.stApp { background: #0a0d14 !important; color: #e2e8f0 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0f1520 0%,#0a0d14 100%) !important;
  border-right: 1px solid #1e2840 !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
  background: linear-gradient(135deg,#111827 60%,#1a2235) !important;
  border: 1px solid #1e2840 !important; border-radius: 16px !important;
  padding: 1.2rem !important; box-shadow: 0 4px 24px rgba(0,0,0,.4);
}
div[data-testid="metric-container"] label { color:#94a3b8!important; font-size:.8rem!important; }
div[data-testid="metric-container"] [data-testid="metric-value"] { color:#f8fafc!important; font-weight:700!important; }
div[data-testid="metric-container"] [data-testid="metric-delta"] { font-size:.78rem!important; }

/* Headers */
h1 { background:linear-gradient(90deg,#6366f1,#8b5cf6,#a78bfa);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent;
     font-weight:800!important; }
h2 { color:#e2e8f0!important; font-weight:700!important; }
h3 { color:#cbd5e1!important; font-weight:600!important; }
p, li, span { color:#cbd5e1; }

/* Buttons */
.stButton > button {
  background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
  color:white!important; border:none!important; border-radius:10px!important;
  font-weight:600!important; transition:transform .15s, box-shadow .15s;
}
.stButton > button:hover {
  transform:translateY(-2px)!important;
  box-shadow:0 8px 24px rgba(99,102,241,.4)!important;
}

/* Inputs */
.stTextInput input, .stSelectbox > div > div {
  background:#111827!important; border:1px solid #1e2840!important;
  border-radius:10px!important; color:#e2e8f0!important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#111827!important; border-radius:12px!important; padding:4px!important; }
.stTabs [data-baseweb="tab"] { color:#94a3b8!important; font-weight:500!important; border-radius:8px!important; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#6366f1,#8b5cf6)!important; color:white!important; }

/* Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#6366f1!important; }

/* Progress bars */
.stProgress > div > div { background:linear-gradient(90deg,#6366f1,#a78bfa)!important; border-radius:99px!important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius:12px!important; overflow:hidden; }
.dvn-scroller { background:#111827!important; }

/* Horizontal rule */
hr { border-color:#1e2840!important; }

/* Alert/info/warning */
.stAlert { border-radius:12px!important; }

/* Plotly charts transparent bg */
.js-plotly-plot .plotly .bg { fill:transparent!important; }

/* Cards */
.card {
  background:linear-gradient(135deg,#111827 60%,#1a2235);
  border:1px solid #1e2840; border-radius:16px;
  padding:1.5rem; margin-bottom:1rem;
  box-shadow:0 4px 24px rgba(0,0,0,.3);
}

/* Risk badges */
.badge-high { background:rgba(239,68,68,.15); color:#f87171; border:1px solid #ef4444;
              border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:600; }
.badge-med  { background:rgba(245,158,11,.15); color:#fbbf24; border:1px solid #f59e0b;
              border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:600; }
.badge-low  { background:rgba(34,197,94,.15);  color:#4ade80; border:1px solid #22c55e;
              border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:600; }

/* Sidebar brand */
.brand-logo { text-align:center; padding:1.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown("""
<div class="brand-logo">
  <div style='font-size:2.5rem;'>🛡️</div>
  <div style='font-size:1.1rem;font-weight:800;color:#a78bfa;letter-spacing:1px;'>ChurnShield</div>
  <div style='font-size:.7rem;color:#4a5568;margin-top:2px;'>Customer Intelligence Platform</div>
</div>
<hr style='border-color:#1e2840;margin:.5rem 0 1rem;'/>
""", unsafe_allow_html=True)
