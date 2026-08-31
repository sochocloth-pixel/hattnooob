"""
CropGuard – AI-Based Crop Health Prediction System
Modern SaaS-style UI with Dark/Light mode, animations & interactivity

Libraries: Pandas, NumPy, Matplotlib, Scikit-learn, Streamlit
CBSE Class 12 level – Python only
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CropGuard | Crop Health Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- THEME STATE --------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# -------------------- LOAD & TRAIN MODEL (cached) --------------------
@st.cache_resource
def load_and_train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "crop_health_data.csv")

    if not os.path.exists(csv_path):
        st.error("Dataset file 'crop_health_data.csv' not found.")
        st.stop()

    df = pd.read_csv(csv_path).dropna()
    expected = ["Humidity", "Temperature", "Rainfall", "Crop_Health"]
    if list(df.columns) != expected:
        if len(df.columns) == 4:
            df.columns = expected
        else:
            st.error("CSV columns invalid")
            st.stop()

    X = df[["Humidity", "Temperature", "Rainfall"]]
    y = df["Crop_Health"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, df


model, accuracy, df = load_and_train_model()

# -------------------- THEME TOGGLE (sidebar top) --------------------
with st.sidebar:
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown("### 🌱 CropGuard")
    with col_t2:
        theme_btn = st.button(
            "🌙" if st.session_state.theme == "light" else "☀️",
            help="Toggle Dark / Light mode",
            key="theme_toggle"
        )
        if theme_btn:
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()

is_dark = st.session_state.theme == "dark"

# -------------------- DYNAMIC CSS (Light + Dark + Animations) --------------------
if is_dark:
    css_vars = """
    :root {
        --bg: #0F1410;
        --bg2: #161C18;
        --card: #1A221C;
        --card-border: #2A352C;
        --text: #E8EDE9;
        --text-muted: #8A9A8C;
        --primary: #3DDC84;
        --primary-dark: #2E7D4F;
        --primary-soft: rgba(61, 220, 132, 0.12);
        --accent: #F9A825;
        --danger: #EF5350;
        --hero-from: #0A1F14;
        --hero-to: #143D2A;
        --shadow: 0 8px 32px rgba(0,0,0,0.4);
        --input-bg: #121812;
        --chip-bg: #1E2A22;
        --chip-border: #2A352C;
        --placeholder-border: #2A352C;
        --step-bg: #1A221C;
    }
    """
else:
    css_vars = """
    :root {
        --bg: #F4F7F2;
        --bg2: #E8F0E6;
        --card: #FFFFFF;
        --card-border: #E0E8E0;
        --text: #0D3B2E;
        --text-muted: #6B7C6E;
        --primary: #1B5E3B;
        --primary-dark: #0D3B2E;
        --primary-soft: rgba(27, 94, 59, 0.08);
        --accent: #F9A825;
        --danger: #C62828;
        --hero-from: #0D3B2E;
        --hero-to: #2E7D4F;
        --shadow: 0 8px 32px rgba(13, 59, 46, 0.1);
        --input-bg: #FAFCFA;
        --chip-bg: #F0F4F0;
        --chip-border: #D4E0D4;
        --placeholder-border: #D4E0D4;
        --step-bg: #F8FAF7;
    }
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

{css_vars}

/* ===== KEYFRAME ANIMATIONS ===== */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(18px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes scaleIn {{
    from {{ opacity: 0; transform: scale(0.92); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}
@keyframes pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(61, 220, 132, 0.35); }}
    50%      {{ box-shadow: 0 0 0 12px rgba(61, 220, 132, 0); }}
}}
@keyframes pulseOrange {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(249, 168, 37, 0.35); }}
    50%      {{ box-shadow: 0 0 0 12px rgba(249, 168, 37, 0); }}
}}
@keyframes pulseRed {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(239, 83, 80, 0.35); }}
    50%      {{ box-shadow: 0 0 0 12px rgba(239, 83, 80, 0); }}
}}
@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes float {{
    0%, 100% {{ transform: translateY(0); }}
    50%      {{ transform: translateY(-4px); }}
}}

/* ===== GLOBAL ===== */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
.stApp {{
    background: linear-gradient(165deg, var(--bg) 0%, var(--bg2) 50%, var(--bg) 100%) !important;
    color: var(--text);
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1180px;
    animation: fadeIn 0.4s ease;
}}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {{
    background: var(--card) !important;
    border-right: 1px solid var(--card-border) !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.2rem;
}}
section[data-testid="stSidebar"] label {{
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}}
section[data-testid="stSidebar"] h3 {{
    color: var(--text) !important;
}}

/* ===== HERO ===== */
.hero {{
    background: linear-gradient(135deg, var(--hero-from) 0%, var(--hero-to) 100%);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease;
}}
.hero::after {{
    content: '';
    position: absolute;
    top: -50%;
    right: -5%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}}
.hero-title {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    position: relative;
    z-index: 1;
}}
.hero-sub {{
    font-size: 0.98rem;
    opacity: 0.88;
    margin: 0.4rem 0 0 0;
    position: relative;
    z-index: 1;
}}
.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.14);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 50px;
    padding: 0.25rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.9rem;
    position: relative;
    z-index: 1;
    animation: pulse 2.5s infinite;
}}

/* ===== RESULT BOXES ===== */
.result-box {{
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    margin: 0.3rem 0 1rem 0;
    animation: scaleIn 0.45s cubic-bezier(0.34, 1.56, 0.64, 1);
}}
.result-healthy {{
    background: linear-gradient(135deg, rgba(46,125,79,0.15) 0%, rgba(61,220,132,0.2) 100%);
    border: 1px solid rgba(46,125,79,0.35);
    animation: scaleIn 0.45s cubic-bezier(0.34, 1.56, 0.64, 1), pulse 2s infinite 0.5s;
}}
.result-moderate {{
    background: linear-gradient(135deg, rgba(249,168,37,0.12) 0%, rgba(249,168,37,0.22) 100%);
    border: 1px solid rgba(249,168,37,0.4);
    animation: scaleIn 0.45s cubic-bezier(0.34, 1.56, 0.64, 1), pulseOrange 2s infinite 0.5s;
}}
.result-poor {{
    background: linear-gradient(135deg, rgba(239,83,80,0.12) 0%, rgba(239,83,80,0.22) 100%);
    border: 1px solid rgba(239,83,80,0.4);
    animation: scaleIn 0.45s cubic-bezier(0.34, 1.56, 0.64, 1), pulseRed 2s infinite 0.5s;
}}
.result-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    opacity: 0.65;
    margin: 0;
    color: var(--text);
}}
.result-status {{
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0.2rem 0 0.45rem 0;
    letter-spacing: -0.4px;
}}
.result-desc {{
    font-size: 0.92rem;
    opacity: 0.85;
    margin: 0;
    line-height: 1.45;
    color: var(--text);
}}

/* ===== CHIPS ===== */
.chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.6rem 0 1rem 0;
    animation: slideInLeft 0.5s ease 0.15s both;
}}
.chip {{
    background: var(--chip-bg);
    border-radius: 50px;
    padding: 0.32rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--primary);
    border: 1px solid var(--chip-border);
    transition: transform 0.2s ease, background 0.2s ease;
}}
.chip:hover {{
    transform: scale(1.06);
}}

/* ===== RECOMMENDATION ===== */
.rec-box {{
    background: var(--card);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border-left: 4px solid var(--primary);
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
    animation: fadeInUp 0.5s ease 0.2s both;
}}
.rec-title {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--primary);
    margin: 0 0 0.25rem 0;
}}
.rec-text {{
    font-size: 0.9rem;
    color: var(--text);
    margin: 0;
    line-height: 1.45;
}}

/* ===== PLACEHOLDER ===== */
.placeholder {{
    background: var(--card);
    border: 2px dashed var(--placeholder-border);
    border-radius: 16px;
    padding: 2.8rem 1.5rem;
    text-align: center;
    color: var(--text-muted);
    animation: fadeIn 0.5s ease;
    transition: border-color 0.3s ease;
}}
.placeholder:hover {{
    border-color: var(--primary);
}}
.placeholder-icon {{
    font-size: 2.6rem;
    margin-bottom: 0.5rem;
    animation: float 3s ease-in-out infinite;
}}

/* ===== SECTION HEADERS ===== */
.section-h {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.7rem 0;
    letter-spacing: -0.3px;
    animation: fadeInUp 0.4s ease;
}}

/* ===== STAT GRID ===== */
.stat-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.45rem;
    margin-top: 0.6rem;
}}
.stat-item {{
    background: var(--primary-soft);
    border-radius: 10px;
    padding: 0.65rem 0.3rem;
    text-align: center;
    border: 1px solid var(--card-border);
    transition: transform 0.2s ease;
}}
.stat-item:hover {{
    transform: scale(1.05);
}}
.stat-value {{
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1.2;
}}
.stat-label {{
    font-size: 0.6rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.1rem;
}}

/* ===== STEPS ===== */
.steps-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.7rem;
}}
.step-card {{
    background: var(--step-bg);
    border-radius: 12px;
    padding: 1rem 0.8rem;
    border: 1px solid var(--card-border);
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.5s ease both;
}}
.step-card:nth-child(1) {{ animation-delay: 0.05s; }}
.step-card:nth-child(2) {{ animation-delay: 0.1s; }}
.step-card:nth-child(3) {{ animation-delay: 0.15s; }}
.step-card:nth-child(4) {{ animation-delay: 0.2s; }}
.step-card:nth-child(5) {{ animation-delay: 0.25s; }}
.step-card:nth-child(6) {{ animation-delay: 0.3s; }}
.step-card:hover {{
    transform: translateY(-4px) scale(1.03);
    box-shadow: var(--shadow);
}}
.step-num {{
    width: 28px;
    height: 28px;
    background: var(--primary);
    color: white;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}}
.step-text {{
    font-size: 0.75rem;
    color: var(--text-muted);
    line-height: 1.35;
    margin: 0;
}}

/* ===== BUTTONS ===== */
.stButton > button {{
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 4px 18px rgba(27, 94, 59, 0.3) !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 28px rgba(27, 94, 59, 0.4) !important;
}}
.stButton > button:active {{
    transform: scale(0.97) !important;
}}

/* ===== SLIDERS ===== */
.stSlider > div > div > div > div {{
    background: var(--primary) !important;
}}

/* ===== FOOTER ===== */
.footer-bar {{
    text-align: center;
    padding: 1.8rem 1rem 0.5rem;
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 500;
    animation: fadeIn 0.8s ease;
}}
.footer-bar span {{
    color: var(--primary);
    font-weight: 700;
}}

/* ===== LIVE INDICATOR ===== */
.live-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #3DDC84;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}}
</style>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR INPUTS (Interactive Sliders) --------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("##### 🌤 Environmental Inputs")
    st.caption("Drag sliders or type exact values")

    temperature = st.slider(
        "🌡 Temperature (°C)",
        min_value=5.0, max_value=45.0,
        value=28.0, step=0.5,
        help="Ideal crop range ≈ 20–32 °C"
    )
    humidity = st.slider(
        "💧 Humidity (%)",
        min_value=10.0, max_value=100.0,
        value=65.0, step=1.0,
        help="Ideal range ≈ 50–80 %"
    )
    rainfall = st.slider(
        "🌧 Rainfall (mm)",
        min_value=0.0, max_value=300.0,
        value=120.0, step=5.0,
        help="Ideal range ≈ 80–180 mm"
    )

    # Live preview chips
    st.markdown(f"""
    <div class="chip-row" style="margin-top:0.4rem;">
        <span class="chip">{temperature:.0f} °C</span>
        <span class="chip">{humidity:.0f} %</span>
        <span class="chip">{rainfall:.0f} mm</span>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("🌿  Predict Crop Health", use_container_width=True)

    # Auto-predict toggle for interactivity
    auto_predict = st.toggle("⚡ Live predict (auto)", value=False, help="Predict automatically when you move sliders")

    st.markdown("---")
    st.markdown("##### 📊 Model Info")
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-item">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Records</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{accuracy*100:.0f}%</div>
            <div class="stat-label">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Decision Tree · max_depth = 6")


# -------------------- PREDICTION LOGIC --------------------
should_predict = predict_btn or auto_predict

if should_predict:
    input_df = pd.DataFrame(
        [[humidity, temperature, rainfall]],
        columns=["Humidity", "Temperature", "Rainfall"]
    )
    prediction = model.predict(input_df)[0]
    st.session_state.predicted = True
    st.session_state.last_pred = prediction
    st.session_state.last_inputs = (temperature, humidity, rainfall)
else:
    prediction = st.session_state.last_pred if st.session_state.predicted else None


# -------------------- HERO --------------------
st.markdown(f"""
<div class="hero">
    <div class="hero-title">🌱 CropGuard</div>
    <div class="hero-sub">Predict crop health from temperature, humidity & rainfall using machine learning</div>
    <div class="hero-badge"><span class="live-dot"></span>Model Ready &nbsp;·&nbsp; Decision Tree &nbsp;·&nbsp; {st.session_state.theme.title()} Mode</div>
</div>
""", unsafe_allow_html=True)


# -------------------- MAIN COLUMNS --------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<p class="section-h">Crop Health Status</p>', unsafe_allow_html=True)

    if prediction:
        if prediction == "Healthy":
            status_color = "#2E7D4F" if not is_dark else "#3DDC84"
            box_class = "result-healthy"
            emoji = "🟢"
            desc = "The current environmental conditions are favorable for crop growth."
            rec = "Conditions appear favourable. Continue regular irrigation and crop monitoring."
        elif prediction == "Moderately Healthy":
            status_color = "#E65100" if not is_dark else "#FFB74D"
            box_class = "result-moderate"
            emoji = "🟡"
            desc = "Environmental conditions are somewhat suitable for the crop."
            rec = "Monitor soil moisture and temperature regularly. Adjust irrigation if needed."
        else:
            status_color = "#C62828" if not is_dark else "#EF5350"
            box_class = "result-poor"
            emoji = "🔴"
            desc = "Conditions may be unfavourable for healthy crop growth."
            rec = "Check irrigation, temperature stress and rainfall conditions carefully."

        t, h, r = st.session_state.last_inputs

        st.markdown(f"""
        <div class="result-box {box_class}">
            <p class="result-label">Prediction Result</p>
            <p class="result-status" style="color:{status_color};">{emoji} {prediction.upper()}</p>
            <p class="result-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="chip-row">
            <span class="chip">🌡 {t} °C</span>
            <span class="chip">💧 {h} %</span>
            <span class="chip">🌧 {r} mm</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rec-box">
            <p class="rec-title">💡 Recommendation</p>
            <p class="rec-text">{rec}</p>
        </div>
        """, unsafe_allow_html=True)

        # Interactive bar chart
        st.markdown('<p class="section-h">Current Environmental Conditions</p>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(5.6, 2.9), facecolor="none")
        bg = "#1A221C" if is_dark else "#FAFCFA"
        text_c = "#E8EDE9" if is_dark else "#555555"
        spine_c = "#2A352C" if is_dark else "#DDDDDD"

        labels = ["Temperature\n(°C)", "Humidity\n(%)", "Rainfall\n(mm)"]
        values = [t, h, r]
        colors = ["#1B5E3B", "#43A047", "#F9A825"]
        bars = ax1.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
        ax1.set_ylabel("Value", fontsize=9, color=text_c)
        ax1.set_ylim(0, max(values) * 1.32 + 5)
        ax1.tick_params(colors=text_c, labelsize=9)
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.03,
                f"{val}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#3DDC84" if is_dark else "#1B5E3B"
            )
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_color(spine_c)
        ax1.spines["bottom"].set_color(spine_c)
        ax1.set_facecolor(bg)
        fig1.patch.set_alpha(0)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">🌿</div>
            <p style="font-weight:600; margin:0 0 0.3rem 0;">Ready to predict</p>
            <p style="margin:0; font-size:0.86rem;">Move the sliders or click <b>Predict</b><br>Enable <b>Live predict</b> for real-time results</p>
        </div>
        """, unsafe_allow_html=True)


with right:
    st.markdown('<p class="section-h">Dataset Overview</p>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(5.3, 3.6), facecolor="none")
    bg = "#1A221C" if is_dark else "#FAFCFA"
    text_c = "#E8EDE9" if is_dark else "#555555"
    spine_c = "#2A352C" if is_dark else "#DDDDDD"
    title_c = "#3DDC84" if is_dark else "#0D3B2E"

    colour_map = {
        "Healthy": "#2E7D4F",
        "Moderately Healthy": "#F9A825",
        "Poor Health": "#E53935"
    }
    for label, colour in colour_map.items():
        subset = df[df["Crop_Health"] == label]
        ax2.scatter(
            subset["Temperature"], subset["Humidity"],
            c=colour, label=label, alpha=0.78, s=34,
            edgecolors="white", linewidths=0.4
        )
    ax2.set_xlabel("Temperature (°C)", fontsize=9, color=text_c)
    ax2.set_ylabel("Humidity (%)", fontsize=9, color=text_c)
    ax2.tick_params(colors=text_c, labelsize=8)
    leg = ax2.legend(fontsize=7.5, loc="best", framealpha=0.9)
    leg.get_frame().set_edgecolor(spine_c)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(spine_c)
    ax2.spines["bottom"].set_color(spine_c)
    ax2.set_facecolor(bg)
    ax2.set_title("Temperature vs Humidity", fontsize=11, fontweight="bold", color=title_c, pad=8)
    fig2.patch.set_alpha(0)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown('<p class="section-h" style="margin-top:0.6rem;">Class Distribution</p>', unsafe_allow_html=True)
    class_counts = df["Crop_Health"].value_counts()
    chart_color = "#3DDC84" if is_dark else "#2E7D4F"
    st.bar_chart(class_counts, color=chart_color, height=170)


# -------------------- HOW IT WORKS --------------------
st.markdown("---")
st.markdown('<p class="section-h">How It Works</p>', unsafe_allow_html=True)

st.markdown("""
<div class="steps-grid">
    <div class="step-card">
        <div class="step-num">1</div>
        <p class="step-text">Load dataset with <b>Pandas</b></p>
    </div>
    <div class="step-card">
        <div class="step-num">2</div>
        <p class="step-text">Select features: Temp, Humidity, Rainfall</p>
    </div>
    <div class="step-card">
        <div class="step-num">3</div>
        <p class="step-text">Split into train & test sets</p>
    </div>
    <div class="step-card">
        <div class="step-num">4</div>
        <p class="step-text">Train <b>Decision Tree</b> model</p>
    </div>
    <div class="step-card">
        <div class="step-num">5</div>
        <p class="step-text">User moves sliders → Predict</p>
    </div>
    <div class="step-card">
        <div class="step-num">6</div>
        <p class="step-text">Show result + <b>Matplotlib</b> charts</p>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer-bar">
    <span>CropGuard</span> · CBSE Class 12 Computer Science Project<br>
    Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
