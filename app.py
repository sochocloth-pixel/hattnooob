"""
CropGuard - Crop Health Prediction System
CBSE Class 12 | No sidebar · Light/Dark · Live tracking
Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="CropGuard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
<style>
section[data-testid="stSidebar"] { display: none !important; }
button[kind="header"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Session ----------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "live" not in st.session_state:
    st.session_state.live = True
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# ---------- Model ----------
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_health_data.csv")
    df = pd.read_csv(path).dropna()
    if len(df.columns) == 4:
        df.columns = ["Humidity", "Temperature", "Rainfall", "Crop_Health"]
    X = df[["Humidity", "Temperature", "Rainfall"]]
    y = df["Crop_Health"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = DecisionTreeClassifier(max_depth=6, random_state=42)
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    return m, acc, df

model, accuracy, df = load_model()
dark = st.session_state.theme == "Dark"

# ---------- Colours ----------
if dark:
    page_bg = "#0F1410"
    card_bg = "#1A221C"
    primary = "#3DDC84"
    text_col = "#E8EDE9"
    muted_col = "#A0B0A4"
    border = "#2A352C"
    chart_bg = "#1A221C"
    chart_text = "#C8D4CC"
    chart_spine = "#3A453C"
    sc = {"Healthy": "#3DDC84", "Moderately Healthy": "#F9A825", "Poor Health": "#EF5350"}
    bar_c = ["#2E7D4F", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg,#0A1F14,#1B5E3B)"
    input_bg = "#121812"
    input_text = "#E8EDE9"
    btn_fg = "#0D3B2E"
else:
    page_bg = "#EEF2EE"
    card_bg = "#FFFFFF"
    primary = "#1B5E3B"
    text_col = "#0A2E1F"
    muted_col = "#3D4F40"
    border = "#C5D0C5"
    chart_bg = "#FFFFFF"
    chart_text = "#1A1A1A"
    chart_spine = "#BBBBBB"
    sc = {"Healthy": "#2E7D4F", "Moderately Healthy": "#F9A825", "Poor Health": "#C62828"}
    bar_c = ["#1B5E3B", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg,#0D3B2E,#1B5E3B)"
    input_bg = "#FFFFFF"
    input_text = "#0A2E1F"
    btn_fg = "#FFFFFF"

# ---------- CSS ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-color: {page_bg} !important; }}
.block-container {{ padding-top: 1rem !important; max-width: 1100px; }}

h1, h2, h3, h4, p, span, label, li, div {{
    color: {text_col} !important;
}}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {{
    color: {text_col} !important;
}}
[data-testid="stWidgetLabel"] p {{
    color: {text_col} !important;
    font-weight: 600 !important;
}}
.stCaption, [data-testid="stCaptionContainer"] p {{
    color: {muted_col} !important;
}}

input[type="number"] {{
    background-color: {input_bg} !important;
    color: {input_text} !important;
    border: 2px solid {border} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
input[type="number"]:focus {{
    border-color: {primary} !important;
    box-shadow: 0 0 0 2px {primary}33 !important;
}}

.stButton > button {{
    background-color: {primary} !important;
    color: {btn_fg} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.25s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 18px {primary}50 !important;
}}
.stButton > button:active {{
    transform: scale(0.97) !important;
}}

div[role="radiogroup"] label {{
    color: {text_col} !important;
    transition: all 0.2s ease !important;
}}
div[role="radiogroup"] label:hover {{
    transform: scale(1.05) !important;
}}

@keyframes livePulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}
.live-dot {{
    display: inline-block;
    width: 9px; height: 9px;
    background: #E53935;
    border-radius: 50%;
    margin-right: 6px;
    animation: livePulse 1.2s ease-in-out infinite;
}}
</style>
""", unsafe_allow_html=True)

# ========== HERO + THEME ==========
top1, top2 = st.columns([4, 1])
with top1:
    live_badge = ' · <span class="live-dot"></span><b>LIVE</b>' if st.session_state.live else ""
    st.markdown(f"""
    <div style="background:{hero};border-radius:14px;padding:1.3rem 1.6rem;">
      <div style="color:white !important;font-size:1.7rem;font-weight:800;">🌱 CropGuard</div>
      <div style="color:#D0E8D8 !important;font-size:0.95rem;margin-top:0.25rem;">
        Predict crop health using temperature, humidity and rainfall{live_badge}
      </div>
    </div>
    """, unsafe_allow_html=True)
with top2:
    st.markdown(f"<p style='color:{text_col};font-weight:700;margin-bottom:0.2rem;'>Theme</p>", unsafe_allow_html=True)
    theme_choice = st.radio(
        "theme",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio"
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

st.write("")

# ========== INPUTS ==========
st.markdown(f"<h3 style='color:{text_col} !important;'>Environmental Conditions</h3>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.1])
with c1:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=28.0, step=0.5, key="t")
with c2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0, key="h")
with c3:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=120.0, step=5.0, key="r")
with c4:
    st.markdown(f"<p style='color:{text_col};font-weight:600;margin-bottom:0.35rem;'>Options</p>", unsafe_allow_html=True)
    live = st.toggle("Live Tracking", value=st.session_state.live, key="live_toggle")
    st.session_state.live = live
    predict_btn = st.button("🌿 Predict", use_container_width=True, key="go")

st.markdown(
    f"<p style='color:{muted_col};font-size:0.85rem;'>"
    f"<b style='color:{text_col};'>Model:</b> Decision Tree &nbsp;·&nbsp; "
    f"<b style='color:{text_col};'>Records:</b> {len(df)} &nbsp;·&nbsp; "
    f"<b style='color:{text_col};'>Accuracy:</b> {accuracy*100:.1f}%</p>",
    unsafe_allow_html=True
)

st.divider()

# ========== PREDICTION ==========
if live or predict_btn:
    inp = pd.DataFrame(
        [[float(humidity), float(temperature), float(rainfall)]],
        columns=["Humidity", "Temperature", "Rainfall"]
    )
    prediction = model.predict(inp)[0]
    pred_inputs = (float(temperature), float(humidity), float(rainfall))
    st.session_state.last_pred = prediction
    st.session_state.last_inputs = pred_inputs
elif st.session_state.last_pred is not None:
    prediction = st.session_state.last_pred
    pred_inputs = st.session_state.last_inputs
else:
    prediction = None
    pred_inputs = None

left, right = st.columns([1.1, 1])

with left:
    st.markdown(f"<h3 style='color:{text_col} !important;'>Crop Health Status</h3>", unsafe_allow_html=True)

    if prediction is not None:
        t, h, r = pred_inputs

        if prediction == "Healthy":
            st.success("**● HEALTHY**\n\nEnvironmental conditions are favorable for crop growth.")
            st.info("**Recommendation:** Continue regular irrigation and crop monitoring.")
        elif prediction == "Moderately Healthy":
            st.warning("**● MODERATELY HEALTHY**\n\nConditions are somewhat suitable. Monitor closely.")
            st.info("**Recommendation:** Check soil moisture and temperature regularly.")
        else:
            st.error("**● POOR HEALTH**\n\nConditions may be unfavourable for healthy growth.")
            st.info("**Recommendation:** Review irrigation, temperature stress and rainfall.")

        st.markdown(
            f"<p style='color:{text_col} !important;font-size:0.95rem;'>"
            f"<b>Inputs:</b> Temp = <b>{t}°C</b> &nbsp;|&nbsp; Humidity = <b>{h}%</b> &nbsp;|&nbsp; Rainfall = <b>{r} mm</b></p>",
            unsafe_allow_html=True
        )
        if live:
            st.markdown(
                f"<p style='color:{primary} !important;font-size:0.8rem;'>"
                f"<span class='live-dot'></span>Live tracking ON — change values above to update</p>",
                unsafe_allow_html=True
            )

        st.markdown(f"<h3 style='color:{text_col} !important;'>Current Conditions</h3>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5.5, 2.6))
        fig1.patch.set_facecolor(chart_bg)
        ax1.set_facecolor(chart_bg)
        vals = [t, h, r]
        bars = ax1.bar(["Temp (°C)", "Humidity (%)", "Rainfall (mm)"], vals, color=bar_c, width=0.5)
        ax1.set_ylim(0, max(vals) * 1.35 + 5)
        ax1.tick_params(colors=chart_text, labelsize=9)
        ax1.set_ylabel("Value", color=chart_text)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.03,
                     str(v), ha="center", fontweight="bold", color=primary, fontsize=10)
        for s in ax1.spines.values():
            s.set_color(chart_spine)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
    else:
        st.info("Enable **Live Tracking** or click **Predict**.")

with right:
    st.markdown(f"<h3 style='color:{text_col} !important;'>Dataset: Temperature vs Humidity</h3>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.2))
    fig2.patch.set_facecolor(chart_bg)
    ax2.set_facecolor(chart_bg)
    for lab, col in sc.items():
        sub = df[df["Crop_Health"] == lab]
        ax2.scatter(sub["Temperature"], sub["Humidity"], c=col, label=lab, alpha=0.8, s=28, edgecolors="white", linewidths=0.3)
    ax2.set_xlabel("Temperature (°C)", color=chart_text, fontsize=9)
    ax2.set_ylabel("Humidity (%)", color=chart_text, fontsize=9)
    ax2.tick_params(colors=chart_text, labelsize=8)
    leg = ax2.legend(fontsize=8, loc="best")
    leg.get_frame().set_facecolor(chart_bg)
    for t_ in leg.get_texts():
        t_.set_color(chart_text)
    for s in ax2.spines.values():
        s.set_color(chart_spine)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown(f"<h3 style='color:{text_col} !important;'>Class Distribution</h3>", unsafe_allow_html=True)
    counts = df["Crop_Health"].value_counts()
    order = ["Healthy", "Moderately Healthy", "Poor Health"]
    vals = [int(counts.get(o, 0)) for o in order]
    cols = [sc[o] for o in order]
    fig3, ax3 = plt.subplots(figsize=(5.2, 2.1))
    fig3.patch.set_facecolor(chart_bg)
    ax3.set_facecolor(chart_bg)
    ax3.barh(order, vals, color=cols, height=0.55)
    ax3.tick_params(colors=chart_text, labelsize=8)
    ax3.set_xlabel("Records", color=chart_text, fontsize=9)
    for i, v in enumerate(vals):
        ax3.text(v + 1.5, i, str(v), va="center", fontweight="bold", color=primary, fontsize=9)
    for s in ax3.spines.values():
        s.set_color(chart_spine)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

st.divider()
st.markdown(f"<h3 style='color:{text_col} !important;'>How It Works (CBSE)</h3>", unsafe_allow_html=True)
st.markdown(f"""
<ol style="color:{text_col} !important;">
<li style="color:{text_col} !important;">Dataset is loaded using <b>Pandas</b></li>
<li style="color:{text_col} !important;">Temperature, humidity and rainfall are selected as <b>features</b></li>
<li style="color:{text_col} !important;">Data is split into <b>training (80%)</b> and <b>testing (20%)</b></li>
<li style="color:{text_col} !important;">A <b>Decision Tree</b> is trained with <b>Scikit-learn</b></li>
<li style="color:{text_col} !important;">User enters values (live or manual)</li>
<li style="color:{text_col} !important;">Model predicts crop health</li>
<li style="color:{text_col} !important;"><b>Matplotlib</b> shows the graphs</li>
</ol>
""", unsafe_allow_html=True)

st.markdown(
    f"<p style='color:{muted_col} !important;font-size:0.8rem;text-align:center;'>"
    f"CropGuard · CBSE Class 12 · Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit</p>",
    unsafe_allow_html=True
)
