import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------
# Load trained model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("eye_cnn_model.h5")

model = load_model()

# Update with your dataset classes
class_names = ["Normal", "Glaucoma", "Cataract"]

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Eye Disease Prediction", page_icon="👁️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f9fd;
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #34495e;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<p class='title'>👁️ Eye Disease Prediction System</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a retinal/fundus image and let the AI model predict possible eye disease</p>", unsafe_allow_html=True)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Upload a clear retinal image in JPG/PNG format.")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ------------------------------
# Prediction
# ------------------------------
if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).resize((128,128))
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing image... Please wait"):
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    st.success(f"✅ Prediction: **{class_names[class_idx]}**")
    st.info(f"📊 Confidence: {confidence:.2f}%")

    # Show confidence chart
    st.subheader("Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0], color=['#27ae60','#2980b9','#e74c3c'])
    ax.set_ylabel("Probability")
    st.pyplot(fig)

else:
    st.warning("👆 Please upload an image from the sidebar to start prediction.")
