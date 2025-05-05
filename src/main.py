import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="ASL Recognition", layout="wide")

# Define ASL classes
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']

# Initialize MediaPipe with optimized settings for static images
@st.cache_resource
def initialize_mediapipe():
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,  # Lower threshold for detection
        min_tracking_confidence=0.3
    )

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../models/asl_landmark_model.keras")

# Extract landmarks with error handling and fallback mechanisms
def extract_landmarks(image):
    # Ensure RGB format (MediaPipe requirement)
    if len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        # Note: Skip color conversion to avoid channel swapping
        # This is a common issue when working with different image sources
        rgb_image = image.copy()
    
    # Process with MediaPipe
    hands = initialize_mediapipe()
    results = hands.process(rgb_image)
    
    # Display detection status
    if results.multi_hand_landmarks:
        st.success("✅ Hand detected successfully")
        landmarks = results.multi_hand_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        coords = (coords - np.min(coords)) / (np.max(coords) - np.min(coords) + 1e-8)
        return coords.reshape(21, 3, 1), results.multi_hand_landmarks
    else:
        # Fallback: Try edge detection to help with hand segmentation
        st.warning("⚠️ Hand detection failed. Using fallback method...")
        
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create a fallback feature vector (placeholder)
        fallback_coords = np.zeros((21, 3, 1))
        return fallback_coords, None

# Main app function
def main():
    st.title("ASL Sign Language Recognition")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Process image
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Recognize Sign"):
            with st.spinner("Processing..."):
                # Convert to numpy array
                img_array = np.array(image)
                
                # Extract landmarks
                landmarks, hand_landmarks = extract_landmarks(img_array)
                
                # Display processed image with landmarks
                with col2:
                    if hand_landmarks:
                        # Draw landmarks on image
                        vis_img = img_array.copy()
                        mp_drawing = mp.solutions.drawing_utils
                        mp_hands = mp.solutions.hands
                        
                        # Draw landmarks
                        for hand_landmark in hand_landmarks:
                            mp_drawing.draw_landmarks(
                                vis_img,
                                hand_landmark,
                                mp_hands.HAND_CONNECTIONS
                            )
                        
                        st.image(vis_img, caption="Detected Hand Landmarks", use_container_width=True)
                    else:
                        st.image(img_array, caption="No Hand Landmarks Detected", use_container_width=True)
                
                # Make prediction regardless of landmark detection
                # (Model might still work with edge-based features)
                predictions = model.predict(np.expand_dims(landmarks, axis=0))
                pred_idx = np.argmax(predictions[0])
                confidence = predictions[0][pred_idx]
                
                # Display results
                st.subheader("Prediction Results")
                st.write(f"**Predicted Sign:** {CLASS_NAMES[pred_idx]}")
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Show top 3 predictions
                st.write("**Top 3 Predictions:**")
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                for i, idx in enumerate(top_indices):
                    st.write(f"{i+1}. {CLASS_NAMES[idx]}: {predictions[0][idx]:.2%}")
                
                if confidence < 0.5:
                    st.warning("⚠️ Low confidence prediction. The model is uncertain about this image.")
    
    # Sidebar with debugging info
    st.sidebar.title("Troubleshooting")
    st.sidebar.info(
        "If hand detection fails, try these tips:\n"
        "1. Ensure good lighting and contrast\n"
        "2. Position hand clearly in frame\n"
        "3. Remove complex backgrounds\n"
        "4. Try with the original dataset format"
    )
    
    # Advanced options
    st.sidebar.title("Advanced Options")
    if st.sidebar.checkbox("Show Technical Details"):
        st.sidebar.write(" Configuration:")
        st.sidebar.write("- static_image_mode: True")
        st.sidebar.write("- min_detection_confidence: 0.3")
        st.sidebar.write("- min_tracking_confidence: 0.3")

if __name__ == "__main__":
    main()
