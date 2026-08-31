"""
CropGuard - Full Featured Crop Health Prediction System
CBSE Class 12 | Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit

Features:
- Light/Dark mode, Live tracking
- Prediction + confidence %
- Health score 0-100, alerts, ideal ranges
- Presets, history, export CSV
- Feature importance, confusion matrix
- Compare two scenarios
- Model switch (Decision Tree / Logistic Regression)
- Dataset explorer, tips
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from datetime import datetime

st.set_page_config(page_title="CropGuard", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
section[data-testid="stSidebar"] { display: none !important; }
button[kind="header"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

defaults = {
    "theme": "Light",
    "live": True,
    "last_pred": None,
    "last_inputs": None,
    "last_proba": None,
    "history": [],
    "model_name": "Decision Tree",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def load_everything():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_health_data.csv")
    df = pd.read_csv(path).dropna()
    if len(df.columns) == 4:
        df.columns = ["Humidity", "Temperature", "Rainfall", "Crop_Health"]
    X = df[["Humidity", "Temperature", "Rainfall"]]
    y = df["Crop_Health"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(Xtr, ytr)

    acc_dt = accuracy_score(yte, dt.predict(Xte))
    acc_lr = accuracy_score(yte, lr.predict(Xte))
    cm_dt = confusion_matrix(yte, dt.predict(Xte), labels=["Healthy", "Moderately Healthy", "Poor Health"])
    cm_lr = confusion_matrix(yte, lr.predict(Xte), labels=["Healthy", "Moderately Healthy", "Poor Health"])

    return df, Xtr, Xte, ytr, yte, dt, lr, acc_dt, acc_lr, cm_dt, cm_lr

df, Xtr, Xte, ytr, yte, model_dt, model_lr, acc_dt, acc_lr, cm_dt, cm_lr = load_everything()

if st.session_state.model_name == "Decision Tree":
    model, accuracy, cm = model_dt, acc_dt, cm_dt
    importances = model_dt.feature_importances_
else:
    model, accuracy, cm = model_lr, acc_lr, cm_lr
    importances = np.abs(model_lr.coef_).mean(axis=0)

dark = st.session_state.theme == "Dark"

if dark:
    page_bg, primary = "#0F1410", "#3DDC84"
    text_col, muted_col = "#E8EDE9", "#A0B0A4"
    border, chart_bg = "#2A352C", "#1A221C"
    chart_text, chart_spine = "#C8D4CC", "#3A453C"
    sc = {"Healthy": "#3DDC84", "Moderately Healthy": "#F9A825", "Poor Health": "#EF5350"}
    bar_c = ["#2E7D4F", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg,#0A1F14,#1B5E3B)"
    input_bg, input_text, btn_fg = "#121812", "#E8EDE9", "#0D3B2E"
else:
    page_bg, primary = "#EEF2EE", "#1B5E3B"
    text_col, muted_col = "#0A2E1F", "#3D4F40"
    border, chart_bg = "#C5D0C5", "#FFFFFF"
    chart_text, chart_spine = "#1A1A1A", "#BBBBBB"
    sc = {"Healthy": "#2E7D4F", "Moderately Healthy": "#F9A825", "Poor Health": "#C62828"}
    bar_c = ["#1B5E3B", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg,#0D3B2E,#1B5E3B)"
    input_bg, input_text, btn_fg = "#FFFFFF", "#0A2E1F", "#FFFFFF"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-color: {page_bg} !important; }}
.block-container {{ padding-top: 0.8rem !important; max-width: 1140px; }}
h1,h2,h3,h4,p,span,label,li,div {{ color: {text_col} !important; }}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {{ color: {text_col} !important; }}
[data-testid="stWidgetLabel"] p {{ color: {text_col} !important; font-weight: 600 !important; }}
input[type="number"] {{
    background: {input_bg} !important; color: {input_text} !important;
    border: 2px solid {border} !important; border-radius: 8px !important; font-weight: 600 !important;
}}
.stButton > button {{
    background: {primary} !important; color: {btn_fg} !important;
    border: none !important; border-radius: 10px !important; font-weight: 700 !important;
    transition: all 0.25s ease !important;
}}
.stButton > button:hover {{ transform: translateY(-2px) scale(1.02) !important; }}
@keyframes livePulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.45}} }}
.live-dot {{
    display:inline-block; width:9px; height:9px; background:#E53935;
    border-radius:50%; margin-right:6px; animation: livePulse 1.2s ease-in-out infinite;
}}
</style>
""", unsafe_allow_html=True)

IDEAL = {"Temperature": (20, 32), "Humidity": (50, 80), "Rainfall": (80, 180)}

def ideal_status(name, val):
    lo, hi = IDEAL[name]
    if lo <= val <= hi:
        return "✅ Ideal"
    if val < lo:
        return "⚠️ Low"
    return "⚠️ High"

def health_score(pred, proba_dict, t, h, r):
    base = proba_dict.get("Healthy", 0) * 100
    bonus = 0
    for name, val in [("Temperature", t), ("Humidity", h), ("Rainfall", r)]:
        lo, hi = IDEAL[name]
        if lo <= val <= hi:
            bonus += 5
    score = min(100, max(0, base + bonus))
    if pred == "Poor Health":
        score = min(score, 40)
    elif pred == "Moderately Healthy":
        score = min(max(score, 35), 70)
    return round(score, 1)

def tip_for(pred):
    tips = {
        "Healthy": "Conditions look good. Keep regular irrigation and watch for pests.",
        "Moderately Healthy": "Monitor soil moisture. Avoid extreme heat or waterlogging.",
        "Poor Health": "Check irrigation, heat stress and drainage. Adjust watering if needed.",
    }
    return tips.get(pred, "")

def run_predict(t, h, r):
    inp = pd.DataFrame([[h, t, r]], columns=["Humidity", "Temperature", "Rainfall"])
    pred = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    classes = list(model.classes_)
    proba_dict = {classes[i]: float(proba[i]) for i in range(len(classes))}
    return pred, proba_dict

h1, h2 = st.columns([4, 1])
with h1:
    live_badge = ' · <span class="live-dot"></span><b>LIVE</b>' if st.session_state.live else ""
    st.markdown(f"""
    <div style="background:{hero};border-radius:14px;padding:1.2rem 1.5rem;">
      <div style="color:white !important;font-size:1.65rem;font-weight:800;">🌱 CropGuard Pro</div>
      <div style="color:#D0E8D8 !important;font-size:0.92rem;margin-top:0.2rem;">
        Full crop health prediction system{live_badge}
      </div>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown(f"<p style='color:{text_col};font-weight:700;'>Theme</p>", unsafe_allow_html=True)
    th = st.radio("th", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1,
                  horizontal=True, label_visibility="collapsed", key="theme_radio")
    if th != st.session_state.theme:
        st.session_state.theme = th
        st.rerun()

st.write("")
st.markdown(f"<h3 style='color:{text_col} !important;'>Environmental Conditions</h3>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{muted_col};font-size:0.85rem;'>Quick presets:</p>", unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns(4)
with p1:
    if st.button("🌿 Ideal Day", use_container_width=True):
        st.session_state.t, st.session_state.h, st.session_state.r = 26.0, 65.0, 120.0
        st.rerun()
with p2:
    if st.button("🔥 Hot & Dry", use_container_width=True):
        st.session_state.t, st.session_state.h, st.session_state.r = 38.0, 30.0, 25.0
        st.rerun()
with p3:
    if st.button("🌧️ Heavy Rain", use_container_width=True):
        st.session_state.t, st.session_state.h, st.session_state.r = 22.0, 92.0, 220.0
        st.rerun()
with p4:
    if st.button("❄️ Cold Stress", use_container_width=True):
        st.session_state.t, st.session_state.h, st.session_state.r = 12.0, 45.0, 40.0
        st.rerun()

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.15])
with c1:
    temperature = st.number_input("Temperature (°C)", -10.0, 55.0, st.session_state.get("t", 28.0), 0.5, key="t")
    st.caption(ideal_status("Temperature", temperature))
with c2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, st.session_state.get("h", 65.0), 1.0, key="h")
    st.caption(ideal_status("Humidity", humidity))
with c3:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, st.session_state.get("r", 120.0), 5.0, key="r")
    st.caption(ideal_status("Rainfall", rainfall))
with c4:
    st.markdown(f"<p style='color:{text_col};font-weight:600;'>Options</p>", unsafe_allow_html=True)
    live = st.toggle("Live Tracking", value=st.session_state.live, key="live_toggle")
    st.session_state.live = live
    predict_btn = st.button("🌿 Predict", use_container_width=True, key="go")

mcol1, mcol2 = st.columns([2, 3])
with mcol1:
    model_choice = st.selectbox("ML Model", ["Decision Tree", "Logistic Regression"],
                                index=0 if st.session_state.model_name == "Decision Tree" else 1, key="model_sel")
    if model_choice != st.session_state.model_name:
        st.session_state.model_name = model_choice
        st.rerun()
with mcol2:
    st.markdown(
        f"<p style='color:{muted_col};font-size:0.85rem;margin-top:1.6rem;'>"
        f"<b style='color:{text_col};'>Accuracy:</b> {accuracy*100:.1f}% &nbsp;·&nbsp; "
        f"<b style='color:{text_col};'>Records:</b> {len(df)} &nbsp;·&nbsp; "
        f"<b style='color:{text_col};'>Features:</b> 3</p>",
        unsafe_allow_html=True
    )

st.divider()

if live or predict_btn:
    prediction, proba_dict = run_predict(temperature, humidity, rainfall)
    pred_inputs = (float(temperature), float(humidity), float(rainfall))
    st.session_state.last_pred = prediction
    st.session_state.last_inputs = pred_inputs
    st.session_state.last_proba = proba_dict
    entry = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Temp": pred_inputs[0], "Humidity": pred_inputs[1], "Rainfall": pred_inputs[2],
        "Prediction": prediction, "Confidence": round(max(proba_dict.values()) * 100, 1),
    }
    hist = st.session_state.history
    if not hist or hist[-1]["Temp"] != entry["Temp"] or hist[-1]["Humidity"] != entry["Humidity"] or hist[-1]["Rainfall"] != entry["Rainfall"] or hist[-1]["Prediction"] != entry["Prediction"]:
        hist.append(entry)
        st.session_state.history = hist[-10:]
elif st.session_state.last_pred is not None:
    prediction = st.session_state.last_pred
    pred_inputs = st.session_state.last_inputs
    proba_dict = st.session_state.last_proba or {}
else:
    prediction = None
    pred_inputs = None
    proba_dict = {}

left, right = st.columns([1.15, 1])

with left:
    st.markdown(f"<h3 style='color:{text_col} !important;'>Crop Health Status</h3>", unsafe_allow_html=True)
    if prediction is not None:
        t, h, r = pred_inputs
        conf = max(proba_dict.values()) * 100 if proba_dict else 0
        score = health_score(prediction, proba_dict, t, h, r)

        if prediction == "Poor Health" or any(ideal_status(n, v).startswith("⚠️") for n, v in [("Temperature", t), ("Humidity", h), ("Rainfall", r)]):
            if prediction == "Poor Health":
                st.error("🚨 **Alert:** Conditions may stress the crop. Review irrigation and temperature.")
            else:
                st.warning("⚠️ **Notice:** One or more values are outside the ideal range.")

        if prediction == "Healthy":
            st.success(f"**● HEALTHY**  \nConfidence: **{conf:.1f}%**  ·  Health Score: **{score}/100**")
        elif prediction == "Moderately Healthy":
            st.warning(f"**● MODERATELY HEALTHY**  \nConfidence: **{conf:.1f}%**  ·  Health Score: **{score}/100**")
        else:
            st.error(f"**● POOR HEALTH**  \nConfidence: **{conf:.1f}%**  ·  Health Score: **{score}/100**")

        st.info(f"**Tip:** {tip_for(prediction)}")
        st.markdown(f"<p style='color:{text_col};font-weight:700;'>Class Probabilities</p>", unsafe_allow_html=True)
        for cls in ["Healthy", "Moderately Healthy", "Poor Health"]:
            pct = proba_dict.get(cls, 0) * 100
            st.progress(min(pct / 100, 1.0), text=f"{cls}: {pct:.1f}%")

        st.markdown(f"<p style='color:{text_col};'><b>Inputs:</b> {t}°C &nbsp;|&nbsp; {h}% &nbsp;|&nbsp; {r} mm</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:{muted_col};font-size:0.85rem;'>"
            f"Temp {ideal_status('Temperature', t)} · Humidity {ideal_status('Humidity', h)} · Rainfall {ideal_status('Rainfall', r)}</p>",
            unsafe_allow_html=True
        )

        st.markdown(f"<h4 style='color:{text_col};'>Current Conditions</h4>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5.4, 2.5))
        fig1.patch.set_facecolor(chart_bg); ax1.set_facecolor(chart_bg)
        vals = [t, h, r]
        bars = ax1.bar(["Temp (°C)", "Humidity (%)", "Rainfall (mm)"], vals, color=bar_c, width=0.5)
        ax1.set_ylim(0, max(vals) * 1.35 + 5)
        ax1.tick_params(colors=chart_text, labelsize=8)
        ax1.set_ylabel("Value", color=chart_text, fontsize=8)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.03, str(v),
                     ha="center", fontweight="bold", color=primary, fontsize=9)
        for s in ax1.spines.values(): s.set_color(chart_spine)
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
        fig1.tight_layout(); st.pyplot(fig1, use_container_width=True); plt.close(fig1)

        report = pd.DataFrame([{
            "Temperature": t, "Humidity": h, "Rainfall": r, "Prediction": prediction,
            "Confidence_%": round(conf, 1), "Health_Score": score,
            "Model": st.session_state.model_name, "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        st.download_button("📥 Download Report (CSV)", report.to_csv(index=False),
                           file_name="cropguard_report.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Enable **Live Tracking** or click **Predict**.")

with right:
    st.markdown(f"<h3 style='color:{text_col} !important;'>Dataset Overview</h3>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(5.1, 3.0))
    fig2.patch.set_facecolor(chart_bg); ax2.set_facecolor(chart_bg)
    for lab, col in sc.items():
        sub = df[df["Crop_Health"] == lab]
        ax2.scatter(sub["Temperature"], sub["Humidity"], c=col, label=lab, alpha=0.8, s=26, edgecolors="white", linewidths=0.3)
    ax2.set_xlabel("Temperature (°C)", color=chart_text, fontsize=8)
    ax2.set_ylabel("Humidity (%)", color=chart_text, fontsize=8)
    ax2.tick_params(colors=chart_text, labelsize=7)
    leg = ax2.legend(fontsize=7, loc="best"); leg.get_frame().set_facecolor(chart_bg)
    for t_ in leg.get_texts(): t_.set_color(chart_text)
    for s in ax2.spines.values(): s.set_color(chart_spine)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig2.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    st.markdown(f"<h4 style='color:{text_col};'>Feature Importance</h4>", unsafe_allow_html=True)
    fig_imp, ax_imp = plt.subplots(figsize=(5.1, 1.8))
    fig_imp.patch.set_facecolor(chart_bg); ax_imp.set_facecolor(chart_bg)
    ax_imp.barh(["Humidity", "Temperature", "Rainfall"], importances, color=bar_c, height=0.5)
    ax_imp.tick_params(colors=chart_text, labelsize=8)
    ax_imp.set_xlabel("Importance", color=chart_text, fontsize=8)
    for s in ax_imp.spines.values(): s.set_color(chart_spine)
    ax_imp.spines["top"].set_visible(False); ax_imp.spines["right"].set_visible(False)
    fig_imp.tight_layout(); st.pyplot(fig_imp, use_container_width=True); plt.close(fig_imp)

    st.markdown(f"<h4 style='color:{text_col};'>Class Distribution</h4>", unsafe_allow_html=True)
    counts = df["Crop_Health"].value_counts()
    order = ["Healthy", "Moderately Healthy", "Poor Health"]
    vals = [int(counts.get(o, 0)) for o in order]
    fig3, ax3 = plt.subplots(figsize=(5.1, 1.9))
    fig3.patch.set_facecolor(chart_bg); ax3.set_facecolor(chart_bg)
    ax3.barh(order, vals, color=[sc[o] for o in order], height=0.5)
    ax3.tick_params(colors=chart_text, labelsize=7)
    for i, v in enumerate(vals):
        ax3.text(v + 1, i, str(v), va="center", fontweight="bold", color=primary, fontsize=8)
    for s in ax3.spines.values(): s.set_color(chart_spine)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    fig3.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close(fig3)

st.divider()
st.markdown(f"<h3 style='color:{text_col} !important;'>Compare Two Scenarios</h3>", unsafe_allow_html=True)
cmp1, cmp2 = st.columns(2)
with cmp1:
    st.markdown(f"<p style='color:{text_col};font-weight:700;'>Scenario A</p>", unsafe_allow_html=True)
    ta = st.number_input("Temp A", -10.0, 55.0, 28.0, 0.5, key="ta")
    ha = st.number_input("Humidity A", 0.0, 100.0, 65.0, 1.0, key="ha")
    ra = st.number_input("Rain A", 0.0, 400.0, 120.0, 5.0, key="ra")
with cmp2:
    st.markdown(f"<p style='color:{text_col};font-weight:700;'>Scenario B</p>", unsafe_allow_html=True)
    tb = st.number_input("Temp B", -10.0, 55.0, 38.0, 0.5, key="tb")
    hb = st.number_input("Humidity B", 0.0, 100.0, 30.0, 1.0, key="hb")
    rb = st.number_input("Rain B", 0.0, 400.0, 25.0, 5.0, key="rb")

if st.button("⚖️ Compare Scenarios", use_container_width=True):
    pa, pra = run_predict(ta, ha, ra)
    pb, prb = run_predict(tb, hb, rb)
    sa, sb = health_score(pa, pra, ta, ha, ra), health_score(pb, prb, tb, hb, rb)
    ca, cb = st.columns(2)
    with ca: st.markdown(f"**A → {pa}** (score {sa}, conf {max(pra.values())*100:.0f}%)")
    with cb: st.markdown(f"**B → {pb}** (score {sb}, conf {max(prb.values())*100:.0f}%)")

st.divider()
st.markdown(f"<h3 style='color:{text_col} !important;'>Prediction History (last 10)</h3>", unsafe_allow_html=True)
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    st.download_button("📥 Download History CSV", hist_df.to_csv(index=False),
                       file_name="cropguard_history.csv", mime="text/csv")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("No predictions yet. Use Live Tracking or Predict.")

st.divider()
ex1, ex2 = st.columns(2)
with ex1:
    st.markdown(f"<h3 style='color:{text_col} !important;'>Confusion Matrix ({st.session_state.model_name})</h3>", unsafe_allow_html=True)
    labels = ["Healthy", "Mod. Healthy", "Poor"]
    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
    fig_cm.patch.set_facecolor(chart_bg); ax_cm.set_facecolor(chart_bg)
    ax_cm.imshow(cm, cmap="Greens")
    ax_cm.set_xticks([0, 1, 2]); ax_cm.set_yticks([0, 1, 2])
    ax_cm.set_xticklabels(labels, color=chart_text, fontsize=8)
    ax_cm.set_yticklabels(labels, color=chart_text, fontsize=8)
    ax_cm.set_xlabel("Predicted", color=chart_text, fontsize=9)
    ax_cm.set_ylabel("Actual", color=chart_text, fontsize=9)
    for i in range(3):
        for j in range(3):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="#0A2E1F", fontweight="bold", fontsize=11)
    fig_cm.tight_layout(); st.pyplot(fig_cm, use_container_width=True); plt.close(fig_cm)

with ex2:
    st.markdown(f"<h3 style='color:{text_col} !important;'>Dataset Explorer</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    st.markdown(
        f"<p style='color:{text_col};font-size:0.85rem;'>"
        f"<b>Temp</b> mean={df['Temperature'].mean():.1f} min={df['Temperature'].min():.1f} max={df['Temperature'].max():.1f}<br>"
        f"<b>Humidity</b> mean={df['Humidity'].mean():.1f} min={df['Humidity'].min():.1f} max={df['Humidity'].max():.1f}<br>"
        f"<b>Rainfall</b> mean={df['Rainfall'].mean():.1f} min={df['Rainfall'].min():.1f} max={df['Rainfall'].max():.1f}</p>",
        unsafe_allow_html=True
    )

st.divider()
st.markdown(f"<h3 style='color:{text_col} !important;'>How It Works (CBSE)</h3>", unsafe_allow_html=True)
st.markdown(f"""
<ol style="color:{text_col} !important;">
<li style="color:{text_col} !important;">Dataset loaded with <b>Pandas</b></li>
<li style="color:{text_col} !important;">Features: Temperature, Humidity, Rainfall</li>
<li style="color:{text_col} !important;">Train/test split 80/20</li>
<li style="color:{text_col} !important;"><b>Decision Tree</b> or <b>Logistic Regression</b> (Scikit-learn)</li>
<li style="color:{text_col} !important;">Predict class + <b>probability</b> (confidence)</li>
<li style="color:{text_col} !important;">Health score, alerts, history, export</li>
<li style="color:{text_col} !important;"><b>Matplotlib</b> charts + confusion matrix</li>
</ol>
""", unsafe_allow_html=True)

st.markdown(
    f"<p style='color:{muted_col} !important;text-align:center;font-size:0.8rem;'>"
    f"CropGuard Pro · CBSE Class 12 · Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit</p>",
    unsafe_allow_html=True
)
