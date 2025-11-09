import streamlit as st
import numpy as np
import joblib
import random

# ------------------ LOAD MODEL (only 1 .pkl file) ------------------
system = joblib.load("stress_system.pkl")
model = system["model"]
scaler = system["scaler"]
encoder = system["encoder"]

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="Stress Monitoring System", page_icon="🧠", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(120deg,#7f7fd5,#86a8e7,#91eae4);
}
.card {
    background:#fff;
    padding:25px;
    border-radius:15px;
    width:450px;
    margin:auto;
    box-shadow:0px 4px 20px rgba(0,0,0,0.25);
}
h1{text-align:center;color:white;font-size:38px;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

quotes = ["🌿 Just breathe.", "💪 You can handle this.", "✨ Stay calm, stay positive."]
st.markdown("<h1>Stress Prediction (ML + Vital Analysis)</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center;color:white;font-size:20px;'>{random.choice(quotes)}</p>",
    unsafe_allow_html=True,
)


# ------------------ HEART RATE EVALUATION ------------------
def evaluate_heartbeat(hr):
    if hr < 60:
        return "⚠ Bradycardia (Low HR)"
    elif 60 <= hr <= 100:
        return "🟢 Normal Heartbeat"
    elif 100 < hr <= 120:
        return "🟡 Moderate Stress (HR high)"
    else:
        return "🔴 High Stress (HR too high)"


# ------------------ OXYGEN EVALUATION ------------------
def evaluate_oxygen(spo2):
    if spo2 >= 85:
        return "🟢 Normal Oxygen Saturation (85–100%)"
    elif 85 > spo2 >= 75:
        return "🟡 Mild Hypoxia (Low Oxygen)"
    elif 75 > spo2 >= 65:
        return "🟠 Moderate Hypoxia (Deep breathing required)"
    else:
        return "🔴 Severe Hypoxia / Critical (Medical help needed!)"


# ------------------ ACTIVITY SUGGESTIONS ------------------
def suggestions(stress_type):
    stress_type = stress_type.lower()

    if "bradycardia" in stress_type:
        return [
            "🛑 Sit down and rest",
            "💧 Drink water",
            "🫁 Take slow deep breaths",
            "⚕ If dizziness continues, seek medical help"
        ]

    elif "high" in stress_type:
        return [
            "🧘 Deep Breathing (4s inhale → 4s hold → 6s exhale)",
            "🎧 Listen to calm music",
            "🚶 Take a short walk",
        ]

    elif "moderate" in stress_type:
        return [
            "☀ Go outside for fresh air",
            "🎯 Do a relaxing activity (drawing, writing)",
            "📞 Talk to someone you trust",
        ]

    else:
        return [
            "✅ You are relaxed",
            "💧 Stay hydrated",
            "🙂 Maintain positive routine",
        ]


# ------------------ UI INPUTS ------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📊 Enter Your Vital Readings")

    spo2 = st.number_input("SpO₂ (%)", min_value=40, max_value=100, value=97)
    hr = st.number_input("Heart Rate (BPM)", min_value=30, max_value=200, value=90)

    if st.button("🔍 Predict Stress", use_container_width=True):

        # Vital Status
        heartbeat_status = evaluate_heartbeat(hr)
        oxygen_status = evaluate_oxygen(spo2)

        st.info(f"❤️ Heart Rate Status: **{heartbeat_status}**")
        st.info(f"🫁 Oxygen Status: **{oxygen_status}**")

        # ML Prediction
        scaled = scaler.transform([[spo2, hr]])
        prediction = model.predict(scaled)[0]
        stress_label = encoder.inverse_transform([prediction])[0]

        st.success(f"🤖 ML Predicted Stress Level: **{stress_label}**")

        # Suggestions based on Heart Rate stress
        st.subheader("💡 Suggested Actions:")
        for tip in suggestions(heartbeat_status):
            st.write(f"- {tip}")

    st.markdown("</div>", unsafe_allow_html=True)
