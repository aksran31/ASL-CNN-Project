import streamlit as st
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import os

MODEL_PATH = "../models/asl_custom_cnn.h5"
ALPNAMES_PATH = "alphabetnames.txt"

# Load label/class names
with open(ALPNAMES_PATH, 'r') as f:
    class_labels = [line.strip() for line in f]

st.set_page_config(
    page_title="ASL Recognition App",
    layout="centered",
)

# Load model once
if "model" not in st.session_state:
    if os.path.exists(MODEL_PATH):
        with st.spinner("Loading model..."):
            st.session_state.model = load_model(MODEL_PATH)
        st.success("Model loaded!")
    else:
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()

# Main UI
st.header("ASL Recognition")

uploaded_file = st.file_uploader(
    "Upload ASL hand sign image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file:
    # Simple preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess for model (200x200 RGB)
    img_array = np.array(image.resize((200, 200)))
    
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:  # Remove alpha channel
        img_array = img_array[:,:,:3]
    
    # Normalize and batch dimension
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Predict"):
        # Get prediction
        predictions = st.session_state.model.predict(img_array)
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]
        
        # Show results
        st.subheader("Prediction Results")
        st.success(f"**Sign**: {class_labels[class_idx]} ({confidence:.1%} confidence)")
        
        # Top 3 predictions table
        top_indices = np.argsort(predictions[0])[::-1][:3]
        results = [{
            "Sign": class_labels[i],
            "Confidence": f"{predictions[0][i]:.1%}"
        } for i in top_indices]
        
        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Please upload an image to get started")
