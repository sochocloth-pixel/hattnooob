"""
CropGuard - Crop Health Prediction System
CBSE Class 12 | Pandas, NumPy, Matplotlib, Scikit-learn, Streamlit
Features: Light/Dark mode, Live tracking, Sidebar inputs
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

# ---------- Session state defaults ----------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "live" not in st.session_state:
    st.session_state.live = True

# ---------- Load & train model ----------
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_health_data.csv")
    df = pd.read_csv(path).dropna()
    if len(df.columns) == 4:
        df.columns = ["Humidity", "Temperature", "Rainfall", "Crop_Health"]
    X = df[["Humidity", "Temperature", "Rainfall"]]
    y = df["Crop_Health"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, df

model, accuracy, df = load_model()

# ---------- Theme ----------
dark = st.session_state.theme == "Dark"

if dark:
    page_bg = "#0F1410"
    card_bg = "#1A221C"
    primary = "#3DDC84"
    chart_bg = "#1A221C"
    chart_text = "#C8D4CC"
    chart_spine = "#3A453C"
    sc = {"Healthy": "#3DDC84", "Moderately Healthy": "#F9A825", "Poor Health": "#EF5350"}
    bar_c = ["#2E7D4F", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg, #0A1F14, #1B5E3B)"
    btn_text = "#0D3B2E"
else:
    page_bg = "#F5F7F4"
    card_bg = "#FFFFFF"
    primary = "#1B5E3B"
    chart_bg = "#FFFFFF"
    chart_text = "#222222"
    chart_spine = "#CCCCCC"
    sc = {"Healthy": "#2E7D4F", "Moderately Healthy": "#F9A825", "Poor Health": "#C62828"}
    bar_c = ["#1B5E3B", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg, #0D3B2E, #1B5E3B)"
    btn_text = "#FFFFFF"

st.markdown(f"""
<style>
.stApp {{ background-color: {page_bg} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem !important; max-width: 1100px; }}

section[data-testid="stSidebar"] {{
    background-color: {card_bg} !important;
    border-right: 3px solid {primary} !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
    background-color: {card_bg} !important;
}}

.stButton > button {{
    background-color: {primary} !important;
    color: {btn_text} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}}
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("🌱 CropGuard")
    st.caption("Crop Health Prediction System")

    theme_choice = st.radio(
        "Theme",
        options=["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        key="theme_radio"
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.divider()

    st.subheader("Environmental Inputs")

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0, max_value=55.0, value=28.0, step=0.5
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0, max_value=100.0, value=65.0, step=1.0
    )
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0, max_value=400.0, value=120.0, step=5.0
    )

    st.divider()

    live = st.toggle("🔴 Live Tracking", value=st.session_state.live,
                     help="Auto-predict when you change any value")
    st.session_state.live = live

    if live:
        st.caption("Live mode ON — prediction updates automatically")
    else:
        st.caption("Live mode OFF — click the button to predict")

    predict_btn = st.button("🌿 Predict Crop Health", use_container_width=True)

    st.divider()
    st.subheader("Model Info")
    st.write(f"**Records:** {len(df)}")
    st.write(f"**Features:** 3")
    st.write(f"**Accuracy:** {accuracy * 100:.1f}%")
    st.write("**Algorithm:** Decision Tree")
    st.caption("max_depth = 6")

# ========== PREDICTION LOGIC ==========
should_predict = live or predict_btn

if should_predict:
    inp = pd.DataFrame(
        [[humidity, temperature, rainfall]],
        columns=["Humidity", "Temperature", "Rainfall"]
    )
    prediction = model.predict(inp)[0]
    pred_inputs = (temperature, humidity, rainfall)
else:
    prediction = None
    pred_inputs = None

# ========== MAIN PAGE ==========
st.markdown(f"""
<div style="background:{hero};border-radius:14px;padding:1.4rem 1.8rem;margin-bottom:1.2rem;">
  <h1 style="color:white;margin:0;font-size:1.75rem;">🌱 CropGuard</h1>
  <p style="color:#D0E8D8;margin:0.3rem 0 0 0;font-size:0.95rem;">
    Predict crop health using temperature, humidity and rainfall
    {"&nbsp;·&nbsp; <b>LIVE</b>" if live else ""}
  </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Crop Health Status")

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

        st.write(f"**Inputs:** Temp = **{t}°C**  |  Humidity = **{h}%**  |  Rainfall = **{r} mm**")

        if live:
            st.caption("🔴 Live tracking active — change sidebar values to update")

        st.subheader("Current Conditions")
        fig1, ax1 = plt.subplots(figsize=(5.5, 2.7))
        fig1.patch.set_facecolor(chart_bg)
        ax1.set_facecolor(chart_bg)
        vals = [t, h, r]
        bars = ax1.bar(["Temp (°C)", "Humidity (%)", "Rainfall (mm)"], vals, color=bar_c, width=0.5)
        ax1.set_ylim(0, max(vals) * 1.35 + 5)
        ax1.tick_params(colors=chart_text, labelsize=9)
        ax1.set_ylabel("Value", color=chart_text)
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.03,
                str(v), ha="center", fontweight="bold", color=primary, fontsize=10
            )
        for s in ax1.spines.values():
            s.set_color(chart_spine)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
    else:
        st.info("Enter values in the **sidebar**, then click **Predict** — or turn on **Live Tracking**.")

with col2:
    st.subheader("Dataset: Temperature vs Humidity")
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.3))
    fig2.patch.set_facecolor(chart_bg)
    ax2.set_facecolor(chart_bg)
    for lab, col in sc.items():
        sub = df[df["Crop_Health"] == lab]
        ax2.scatter(
            sub["Temperature"], sub["Humidity"],
            c=col, label=lab, alpha=0.8, s=28,
            edgecolors="white", linewidths=0.3
        )
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

    st.subheader("Class Distribution")
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
st.subheader("How It Works (CBSE)")
st.markdown("""
1. The dataset is loaded using **Pandas**
2. Temperature, humidity and rainfall are selected as **features**
3. Data is split into **training (80%)** and **testing (20%)** sets
4. A **Decision Tree** model is trained with **Scikit-learn**
5. User enters environmental values (live or manual predict)
6. The model predicts the crop health class
7. **Matplotlib** displays the graphs
""")

st.caption("CropGuard · CBSE Class 12 · Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit")
