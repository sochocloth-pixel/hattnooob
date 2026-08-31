"""
CropGuard - Crop Health Prediction System
CBSE Class 12 | Light/Dark mode · Live tracking · Animated toggles
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
    initial_sidebar_state="expanded"
)

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
    page_bg, card_bg, primary = "#0F1410", "#1A221C", "#3DDC84"
    text_col, muted_col = "#E8EDE9", "#A0B0A4"
    chart_bg, chart_text, chart_spine = "#1A221C", "#C8D4CC", "#3A453C"
    sc = {"Healthy": "#3DDC84", "Moderately Healthy": "#F9A825", "Poor Health": "#EF5350"}
    bar_c = ["#2E7D4F", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg,#0A1F14,#1B5E3B)"
    input_bg, input_text = "#121812", "#E8EDE9"
    btn_fg = "#0D3B2E"
else:
    page_bg, card_bg, primary = "#F5F7F4", "#FFFFFF", "#1B5E3B"
    text_col, muted_col = "#0D3B2E", "#4A5C50"
    chart_bg, chart_text, chart_spine = "#FFFFFF", "#222222", "#CCCCCC"
    sc = {"Healthy": "#2E7D4F", "Moderately Healthy": "#F9A825", "Poor Health": "#C62828"}
    bar_c = ["#1B5E3B", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg,#0D3B2E,#1B5E3B)"
    input_bg, input_text = "#FFFFFF", "#0D3B2E"
    btn_fg = "#FFFFFF"

# ---------- CSS ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-color: {page_bg} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem !important; max-width: 1100px; }}

section[data-testid="stSidebar"] {{
    background-color: {card_bg} !important;
    border-right: 3px solid {primary} !important;
}}
section[data-testid="stSidebar"] > div {{
    background-color: {card_bg} !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
    color: {text_col} !important;
}}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {{
    color: {muted_col} !important;
}}
section[data-testid="stSidebar"] input {{
    background-color: {input_bg} !important;
    color: {input_text} !important;
    border: 1.5px solid {primary}40 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}

.main h1, .main h2, .main h3,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {{
    color: {text_col} !important;
}}
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {muted_col} !important;
}}
.stSubheader, [data-testid="stSubheader"] {{
    color: {text_col} !important;
}}

.stButton > button {{
    background-color: {primary} !important;
    color: {btn_fg} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.25s cubic-bezier(0.34, 1.4, 0.64, 1) !important;
    box-shadow: 0 2px 8px {primary}40 !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 20px {primary}55 !important;
}}
.stButton > button:active {{
    transform: scale(0.97) !important;
}}

[data-testid="stWidgetLabel"] p {{
    color: {text_col} !important;
    font-weight: 600 !important;
}}
div[data-baseweb="checkbox"] {{
    transition: all 0.3s ease !important;
}}
div[role="radiogroup"] label {{
    transition: all 0.2s ease !important;
    border-radius: 8px !important;
}}
div[role="radiogroup"] label:hover {{
    transform: scale(1.04) !important;
}}

@keyframes livePulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.6; transform: scale(1.15); }}
}}
.live-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    background: #E53935;
    border-radius: 50%;
    margin-right: 6px;
    animation: livePulse 1.2s ease-in-out infinite;
}}
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown(f"<h2 style='color:{text_col};margin:0 0 0.2rem 0;'>🌱 CropGuard</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{muted_col};font-size:0.85rem;margin:0 0 1rem 0;'>Crop Health Prediction System</p>", unsafe_allow_html=True)

    st.markdown(f"<p style='color:{text_col};font-weight:700;margin-bottom:0.3rem;'>Theme</p>", unsafe_allow_html=True)
    theme_choice = st.radio(
        "theme_select",
        options=["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio"
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.divider()
    st.markdown(f"<p style='color:{text_col};font-weight:700;'>Environmental Inputs</p>", unsafe_allow_html=True)

    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=28.0, step=0.5, key="inp_temp")
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0, key="inp_hum")
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=120.0, step=5.0, key="inp_rain")

    st.divider()

    live = st.toggle("Live Tracking", value=st.session_state.live, key="live_toggle")
    st.session_state.live = live

    if live:
        st.markdown(
            f"<p style='color:{primary};font-size:0.8rem;'><span class='live-dot'></span>Live ON — updates as you change values</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"<p style='color:{muted_col};font-size:0.8rem;'>Live OFF — click Predict button</p>", unsafe_allow_html=True)

    predict_btn = st.button("🌿 Predict Crop Health", use_container_width=True, key="predict_btn")

    st.divider()
    st.markdown(f"<p style='color:{text_col};font-weight:700;'>Model Info</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_col};font-size:0.9rem;'><b>Records:</b> {len(df)}<br><b>Features:</b> 3<br><b>Accuracy:</b> {accuracy*100:.1f}%<br><b>Model:</b> Decision Tree</p>", unsafe_allow_html=True)

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

# ========== MAIN ==========
live_badge = ' · <span class="live-dot"></span><b>LIVE</b>' if live else ""
st.markdown(f"""
<div style="background:{hero};border-radius:14px;padding:1.4rem 1.8rem;margin-bottom:1.2rem;">
  <h1 style="color:white !important;margin:0;font-size:1.75rem;">🌱 CropGuard</h1>
  <p style="color:#D0E8D8 !important;margin:0.3rem 0 0 0;font-size:0.95rem;">
    Predict crop health using temperature, humidity and rainfall{live_badge}
  </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1])

with col1:
    st.markdown(f"<h3 style='color:{text_col};'>Crop Health Status</h3>", unsafe_allow_html=True)

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
            f"<p style='color:{text_col};font-size:0.95rem;'><b>Inputs:</b> Temp = <b>{t}°C</b> &nbsp;|&nbsp; Humidity = <b>{h}%</b> &nbsp;|&nbsp; Rainfall = <b>{r} mm</b></p>",
            unsafe_allow_html=True
        )

        st.markdown(f"<h3 style='color:{text_col};'>Current Conditions</h3>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5.5, 2.7))
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
        st.info("Turn on **Live Tracking** or click **Predict Crop Health** in the sidebar.")

with col2:
    st.markdown(f"<h3 style='color:{text_col};'>Dataset: Temperature vs Humidity</h3>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.3))
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

    st.markdown(f"<h3 style='color:{text_col};'>Class Distribution</h3>", unsafe_allow_html=True)
    counts = df["Crop_Health"].value_counts()
    order = ["Healthy", "Moderately Healthy", "Poor Health"]
    vals = [int(counts.get(o, 0)) for o in order]
    cols = [sc[o] for o in order]
    fig3, ax3 = plt.subplots(figsize=(5.2, 2.2))
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
st.markdown(f"<h3 style='color:{text_col};'>How It Works (CBSE)</h3>", unsafe_allow_html=True)
st.markdown(f"""
<ol style="color:{text_col};">
<li>Dataset is loaded using <b>Pandas</b></li>
<li>Temperature, humidity and rainfall are selected as <b>features</b></li>
<li>Data is split into <b>training (80%)</b> and <b>testing (20%)</b> sets</li>
<li>A <b>Decision Tree</b> model is trained with <b>Scikit-learn</b></li>
<li>User enters values (live tracking or manual predict)</li>
<li>Model predicts the crop health class</li>
<li><b>Matplotlib</b> displays the graphs</li>
</ol>
""", unsafe_allow_html=True)

st.markdown(f"<p style='color:{muted_col};font-size:0.8rem;text-align:center;'>CropGuard · CBSE Class 12 · Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit</p>", unsafe_allow_html=True)
