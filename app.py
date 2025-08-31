import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("eye_disease_model_compressed.h5")

model = load_model()

# -------------------------------
# Helper Function
# -------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize image for prediction"""
    img = image.resize((224, 224))   # Change size based on your model input
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:   # if grayscale → convert to RGB
        img_array = np.stack((img_array,)*3, axis=-1)
    return np.expand_dims(img_array, axis=0)

# Replace with your class labels
CLASS_NAMES = ["Cataract", "Glaucoma", "Normal", "Diabetic Retinopathy"]

def predict(image: Image.Image):
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img)
    confidence = np.max(preds)
    label = CLASS_NAMES[np.argmax(preds)]
    return label, confidence

# -------------------------------
# UI Layout
# -------------------------------
st.title("🧿 Eye Disease Classification App")
st.markdown(
    """
    ### Welcome to the Eye Disease Detection System  
    Upload an **eye image** and our AI model will classify it into possible conditions.  
    This tool is designed to assist doctors and patients for **faster preliminary diagnosis**.  
    """
)

st.sidebar.header("📂 Upload Section")
uploaded_file = st.sidebar.file_uploader("Upload an eye image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(image, caption="Uploaded Eye Image", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Prediction")
        label, confidence = predict(image)
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")

        # Add progress bar
        st.progress(int(confidence*100))

else:
    st.info("👈 Please upload an image from the sidebar to get started.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("👩‍⚕️ Developed as a Deep Learning Project for Eye Disease Detection")
