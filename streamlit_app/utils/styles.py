"""
utils/styles.py
---------------
ChurnShield — Shared theme engine.
Provides dark / light mode toggle via st.session_state['theme'].
Import in EVERY page with:
    from utils.styles import apply_styles, sidebar_brand
    apply_styles(); sidebar_brand()
"""
import streamlit as st


# ── palette ──────────────────────────────────────────────────────────────────
DARK = {
    "bg":           "#080c14",
    "bg2":          "#0e1623",
    "bg3":          "#111d2e",
    "border":       "#1a2540",
    "border2":      "#243050",
    "text":         "#e8edf5",
    "text2":        "#8a9ab5",
    "text3":        "#4a5a7a",
    "accent":       "#5b6ef5",
    "accent2":      "#7c3aed",
    "accent3":      "#06b6d4",
    "success":      "#10b981",
    "warning":      "#f59e0b",
    "danger":       "#ef4444",
    "card_shadow":  "0 4px 32px rgba(0,0,0,0.5)",
    "glow":         "0 0 24px rgba(91,110,245,0.18)",
}

LIGHT = {
    "bg":           "#f0f4fb",
    "bg2":          "#ffffff",
    "bg3":          "#e8eef8",
    "border":       "#d0d9ee",
    "border2":      "#b8c6e0",
    "text":         "#0f1c35",
    "text2":        "#3d5080",
    "text3":        "#8096be",
    "accent":       "#4355e8",
    "accent2":      "#6d28d9",
    "accent3":      "#0891b2",
    "success":      "#059669",
    "warning":      "#d97706",
    "danger":       "#dc2626",
    "card_shadow":  "0 2px 20px rgba(67,85,232,0.10)",
    "glow":         "0 0 24px rgba(67,85,232,0.12)",
}


def _theme() -> dict:
    """Return current palette dict based on session state."""
    return DARK if st.session_state.get("theme", "dark") == "dark" else LIGHT


def apply_styles():
    """Inject all CSS. Call once at the top of every page."""
    t = _theme()
    is_dark = st.session_state.get("theme", "dark") == "dark"

    # toggle button label + icon
    toggle_label = "☀️ Light Mode" if is_dark else "🌙 Dark Mode"

    st.markdown(f"""
<style>
/* ── FONTS ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── RESET & BASE ───────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"], .stApp {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: {t["bg"]} !important;
    color: {t["text"]} !important;
    transition: background-color 0.35s ease, color 0.35s ease;
}}

/* ── HIDE STREAMLIT CHROME ──────────────────────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem !important; max-width: 1300px; }}

/* ── SIDEBAR ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {t["bg2"]} !important;
    border-right: 1px solid {t["border"]} !important;
    padding-top: 0 !important;
}}
section[data-testid="stSidebar"] * {{
    color: {t["text"]} !important;
}}
section[data-testid="stSidebar"] a {{
    border-radius: 10px !important;
    transition: background 0.2s, padding-left 0.2s !important;
}}
section[data-testid="stSidebar"] a:hover {{
    background: {t["bg3"]} !important;
    padding-left: 8px !important;
}}

/* ── TYPOGRAPHY ─────────────────────────────────────────────────── */
h1 {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    background: linear-gradient(135deg, {t["accent"]}, {t["accent2"]}, {t["accent3"]});
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.5px !important;
    line-height: 1.15 !important;
}}
h2 {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: {t["text"]} !important;
    font-size: 1.6rem !important;
}}
h3 {{
    font-weight: 600 !important;
    color: {t["text2"]} !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.02em !important;
}}
p, li, span, div {{ color: {t["text2"]}; }}

/* ── METRIC CARDS ───────────────────────────────────────────────── */
div[data-testid="metric-container"] {{
    background: {t["bg2"]} !important;
    border: 1px solid {t["border"]} !important;
    border-radius: 18px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: {t["card_shadow"]} !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    position: relative;
    overflow: hidden;
}}
div[data-testid="metric-container"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {t["accent"]}, {t["accent3"]});
    border-radius: 18px 18px 0 0;
}}
div[data-testid="metric-container"]:hover {{
    transform: translateY(-3px) !important;
    box-shadow: {t["glow"]} !important;
}}
div[data-testid="metric-container"] label {{
    color: {t["text3"]} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}
div[data-testid="metric-container"] [data-testid="metric-value"] {{
    color: {t["text"]} !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
}}
div[data-testid="metric-container"] [data-testid="metric-delta"] {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}}

/* ── CARDS ──────────────────────────────────────────────────────── */
.cs-card {{
    background: {t["bg2"]};
    border: 1px solid {t["border"]};
    border-radius: 20px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    box-shadow: {t["card_shadow"]};
    transition: transform 0.22s cubic-bezier(.34,1.56,.64,1),
                box-shadow 0.22s ease,
                border-color 0.22s ease;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.4s ease both;
}}
.cs-card:hover {{
    transform: translateY(-5px) scale(1.012);
    box-shadow: {t["glow"]}, {t["card_shadow"]};
    border-color: {t["accent"]};
}}
.cs-card::after {{
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    background: linear-gradient(135deg,
        rgba(91,110,245,0.04) 0%,
        transparent 60%);
    pointer-events: none;
}}
.cs-card-accent-top {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 20px 20px 0 0;
}}
.cs-card-icon {{
    font-size: 2.2rem;
    margin-bottom: 0.75rem;
    display: block;
    filter: drop-shadow(0 2px 8px rgba(91,110,245,0.3));
}}
.cs-card-title {{
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: {t["text"]};
    margin-bottom: 0.4rem;
}}
.cs-card-desc {{
    font-size: 0.83rem;
    color: {t["text3"]};
    line-height: 1.6;
}}

/* ── BUTTONS ────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {t["accent"]}, {t["accent2"]}) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: transform 0.18s, box-shadow 0.18s !important;
    box-shadow: 0 4px 14px rgba(91,110,245,0.3) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(91,110,245,0.45) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* ── INPUTS ─────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input {{
    background: {t["bg3"]} !important;
    border: 1.5px solid {t["border"]} !important;
    border-radius: 12px !important;
    color: {t["text"]} !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    padding: 0.6rem 1rem !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color: {t["accent"]} !important;
    box-shadow: 0 0 0 3px rgba(91,110,245,0.15) !important;
}}
.stSelectbox > div > div {{
    background: {t["bg3"]} !important;
    border: 1.5px solid {t["border"]} !important;
    border-radius: 12px !important;
    color: {t["text"]} !important;
}}

/* ── SLIDERS ────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: {t["accent"]} !important;
    box-shadow: 0 0 0 4px rgba(91,110,245,0.2) !important;
}}
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {{
    color: {t["text3"]} !important;
}}

/* ── TABS ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {t["bg2"]} !important;
    border-radius: 14px !important;
    padding: 5px !important;
    border: 1px solid {t["border"]} !important;
    gap: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: {t["text2"]} !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: background 0.2s !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {t["accent"]}, {t["accent2"]}) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(91,110,245,0.3) !important;
}}

/* ── ALERTS ─────────────────────────────────────────────────────── */
.stAlert {{
    border-radius: 14px !important;
    border-left-width: 4px !important;
}}

/* ── PROGRESS ───────────────────────────────────────────────────── */
.stProgress > div > div {{
    background: linear-gradient(90deg, {t["accent"]}, {t["accent3"]}) !important;
    border-radius: 99px !important;
}}

/* ── DIVIDER ────────────────────────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid {t["border"]} !important;
    margin: 1.5rem 0 !important;
}}

/* ── PLOTLY TRANSPARENT ─────────────────────────────────────────── */
.js-plotly-plot .plotly .bg {{ fill: transparent !important; }}

/* ── DATAFRAME ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{ border-radius: 14px !important; overflow: hidden; }}
.dvn-scroller {{ background: {t["bg2"]} !important; }}

/* ── STAT PILL ──────────────────────────────────────────────────── */
.stat-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: {t["bg3"]};
    border: 1px solid {t["border"]};
    border-radius: 99px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: {t["text2"]};
}}

/* ── RISK BADGES ────────────────────────────────────────────────── */
.badge-high {{
    background: rgba(239,68,68,.15); color: #f87171;
    border: 1px solid rgba(239,68,68,.4);
    border-radius: 20px; padding: 3px 12px;
    font-size: .78rem; font-weight: 700;
}}
.badge-med {{
    background: rgba(245,158,11,.15); color: #fbbf24;
    border: 1px solid rgba(245,158,11,.4);
    border-radius: 20px; padding: 3px 12px;
    font-size: .78rem; font-weight: 700;
}}
.badge-low {{
    background: rgba(16,185,129,.15); color: #34d399;
    border: 1px solid rgba(16,185,129,.4);
    border-radius: 20px; padding: 3px 12px;
    font-size: .78rem; font-weight: 700;
}}

/* ── ANIMATIONS ─────────────────────────────────────────────────── */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0);    }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes pulse-glow {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(91,110,245,0); }}
    50%       {{ box-shadow: 0 0 20px 4px rgba(91,110,245,0.25); }}
}}
.animate-fade {{ animation: fadeIn 0.5s ease both; }}
.animate-up   {{ animation: fadeSlideUp 0.45s ease both; }}

/* ── TOGGLE BUTTON (custom) ─────────────────────────────────────── */
.theme-toggle {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: {t["bg3"]};
    border: 1.5px solid {t["border"]};
    border-radius: 99px;
    padding: 6px 18px;
    cursor: pointer;
    font-size: 0.82rem;
    font-weight: 600;
    color: {t["text2"]};
    transition: all 0.2s;
    margin: 0.5rem 0;
}}
.theme-toggle:hover {{
    border-color: {t["accent"]};
    color: {t["accent"]};
}}

/* ── SIDEBAR NAV LINKS ──────────────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
    padding: 0.5rem 1rem !important;
    border-radius: 10px !important;
    margin: 2px 0 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    transition: background 0.2s, color 0.2s !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
    background: {t["bg3"]} !important;
    color: {t["accent"]} !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] [aria-current="page"] {{
    background: linear-gradient(135deg, rgba(91,110,245,0.18), rgba(124,58,237,0.12)) !important;
    border-left: 3px solid {t["accent"]} !important;
    color: {t["accent"]} !important;
    font-weight: 700 !important;
}}
</style>
""", unsafe_allow_html=True)


def sidebar_brand():
    """Render logo, tagline, theme toggle, and divider in sidebar."""
    t = _theme()
    is_dark = st.session_state.get("theme", "dark") == "dark"
    logo_bg = "rgba(91,110,245,0.15)" if is_dark else "rgba(67,85,232,0.10)"
    subtitle_color = t["text3"]
    version_color = t["text3"]

    st.sidebar.markdown(f"""
<div style="padding: 1.6rem 1rem 0.8rem; animation: fadeSlideUp 0.4s ease;">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <div style="
        width:44px; height:44px;
        background:{logo_bg};
        border-radius:14px;
        display:flex; align-items:center; justify-content:center;
        font-size:1.6rem;
        border:1px solid {t['border']};
        box-shadow: 0 4px 12px rgba(91,110,245,0.2);
    ">🛡️</div>
    <div>
      <div style="
          font-family:'Syne',sans-serif;
          font-weight:800;
          font-size:1.25rem;
          background:linear-gradient(135deg,{t['accent']},{t['accent2']});
          -webkit-background-clip:text;
          -webkit-text-fill-color:transparent;
          background-clip:text;
          letter-spacing:-0.3px;
      ">ChurnShield</div>
      <div style="font-size:0.68rem;color:{subtitle_color};font-weight:500;letter-spacing:0.06em;text-transform:uppercase;">
        Intelligence Platform
      </div>
    </div>
  </div>
  <div style="font-size:0.72rem;color:{version_color};margin-top:6px;padding-left:2px;">
    v2.0 · Telco Churn · India Market
  </div>
</div>
<hr style="border:none;border-top:1px solid {t['border']};margin:0.6rem 1rem 0.8rem;"/>
""", unsafe_allow_html=True)

    # ── Theme toggle via a real st.button ──────────────────────────
    toggle_icon = "☀️" if is_dark else "🌙"
    toggle_text = "Light Mode" if is_dark else "Dark Mode"

    st.sidebar.markdown("<div style='padding: 0 1rem;'>", unsafe_allow_html=True)
    if st.sidebar.button(f"{toggle_icon}  {toggle_text}", key="__theme_toggle__",
                         use_container_width=True):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown(f"""
<hr style="border:none;border-top:1px solid {t['border']};margin:0.8rem 1rem 0.5rem;"/>
<div style="padding:0 1rem;">
  <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
              letter-spacing:0.1em;color:{t['text3']};margin-bottom:0.5rem;">
    Navigation
  </div>
</div>
""", unsafe_allow_html=True)


def card(icon: str, title: str, desc: str, accent_color: str,
         delay: float = 0) -> str:
    """Return HTML for a styled feature card."""
    return f"""
<div class="cs-card" style="animation-delay:{delay}s;">
  <div class="cs-card-accent-top"
       style="background:linear-gradient(90deg,{accent_color},transparent);"></div>
  <span class="cs-card-icon">{icon}</span>
  <div class="cs-card-title">{title}</div>
  <div class="cs-card-desc">{desc}</div>
</div>"""


def section_header(title: str, subtitle: str = ""):
    """Render a consistent section header with optional subtitle."""
    t = _theme()
    sub_html = f'<p style="color:{t["text3"]};font-size:.88rem;margin-top:-4px;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"<h2 style='margin-bottom:2px;'>{title}</h2>{sub_html}", unsafe_allow_html=True)