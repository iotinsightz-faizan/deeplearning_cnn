import streamlit as st
import numpy as np
import joblib
import random

system = joblib.load("stress_system.pkl")
model = system["model"]
scaler = system["scaler"]
encoder = system["encoder"]

st.set_page_config(page_title="Stress Prediction", page_icon="🧠", layout="wide")

# UI styling
st.markdown("""
<style>
body {
    background: linear-gradient(120deg,#7f7fd5,#86a8e7,#91eae4);
}
.card {
    background:#fff;
    padding:25px;
    border-radius:15px;
    width:430px;
    margin:auto;
    box-shadow:0px 4px 20px rgba(0,0,0,0.25);
}
h1{text-align:center;color:white;font-size:38px;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

messages = ["🌿 Just breathe.", "💪 You can handle this.", "✨ Stay calm, stay positive."]
st.markdown("<h1>Stress Prediction (ML Based)</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:white;font-size:20px;'>{random.choice(messages)}</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    spo2 = st.number_input("SpO₂ (%)", min_value=40, max_value=100, value=97)
    hr = st.number_input("Heart Rate (BPM)", min_value=30, max_value=180, value=90)

    if st.button("🔍 Predict Stress", use_container_width=True):

        scaled_input = scaler.transform([[spo2, hr]])
        prediction = model.predict(scaled_input)[0]
        stress_level = encoder.inverse_transform([prediction])[0]

        st.success(f"🤖 Predicted Stress Level: **{stress_level}**")

    st.markdown("</div>", unsafe_allow_html=True)
