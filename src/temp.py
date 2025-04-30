
import streamlit as st
from streamlit_cropper import st_cropper
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

MODEL_PATH = "..models/asl_custom_cnn.h5"
ALPNAMES_PATH = "model/alphabetnames.txt"

# Load model
model = load_model(MODEL_PATH)

# Load label/class names
with open(LABELS_PATH, 'r') as f:
    class_labels = [line.strip() for line in f]

st.set_page_config(
    page_title="Font detection app",
    layout="wide",
)

# MARK: Intro and Tutorial
st.title("🔠 Machine Learning Based Font Detection App")
st.write(
    """
The app allows users to upload an image of a character (screenshot, camera capture, etc.), and it will predict the font used in the image. The model has been trained on a diverse set of fonts to ensure accurate recognition.
    """
)

with st.expander("**How to use the app?**", icon="❓"):
    st.write(
    """
    1. **Upload an Image**:
    Click the **"Upload Image"** button and select an image file containing the text or character you want to analyze. Supported formats: **JPG, PNG, JPEG**.

    2. **Crop the Image**:
    Once the image is uploaded, a cropping dialog will appear. Adjust the crop area to isolate the character you want to detect, and adjust brightness and contrast as necessary. When done, click **"Save Crop"** to store the cropped character.

    3. **Enter the Character**:
    In the text input field, type the character you cropped. **Make sure it exactly matches** (case-sensitive) to ensure accurate predictions.

    4. **Predict the Font**:
    Click the **"Predict"** button to process your input through the model. The app will display the predicted font in the **Output** section.
    """
    )

st.divider()

# MARK: Main content
if "model" not in st.session_state:
    # st.spinner("Model status: loading, please wait a moment...")
    st.warning("Model status: loading, please wait a moment...", icon="⏳")

else:
    st.success("Model status: loaded!", icon="✅")

    st.header("✒️ Inputs")

    # MARK: | Crop dialog
    @st.dialog("🖼️ Crop the image to the character", width="large")
    def crop(img: Image.Image):

        st.write("Use the zoom slider to enlarge the image for your convenience.")
        SCALE = st.slider("Zoom", 1, 5, 2)

        st.write("Adjust the following so that the resulting image is **black text on white background.**")
        CONTRAST = st.slider("Contrast", 0.5, 5.0, 1.0)
        BRIGHTNESS = st.slider("Brightness", 0.0, 2.0, 1.0)

        # THRESHOLD_ACTIVATE = st.checkbox(label="Enable mono threshold?")
        # THRESHOLD_SLIDER = st.slider("Mono threshold", min_value=0, max_value=255, value=127, disabled=not THRESHOLD_ACTIVATE)

        box = st_cropper(
            img.resize((img.width * SCALE, img.height * SCALE)),
            realtime_update=True,
            aspect_ratio=None,
            should_resize_image=True,
            return_type="box",
        )

        # left, upper, right, lower
        final_image = img.crop(
            (
                (box["left"]) / SCALE,
                (box["top"]) / SCALE,
                (box["left"] + box["width"]) / SCALE,
                (box["top"] + box["height"]) / SCALE,
            )
        )

        final_image = final_image.convert("L")
        final_image = ImageEnhance.Contrast(final_image).enhance(CONTRAST)
        final_image = ImageEnhance.Brightness(final_image).enhance(BRIGHTNESS)
        # if THRESHOLD_ACTIVATE:
        #     margin = 20

        #     map = list(range(256))
        #     map[:max(0, THRESHOLD_SLIDER-margin)] = [0] * max(0, THRESHOLD_SLIDER-margin)
        #     map[max(0, THRESHOLD_SLIDER-margin):min(THRESHOLD_SLIDER+margin, 255)] = np.linspace(0, 255, 2*margin, endpoint=True, dtype=np.uint8)
        #     map[min(THRESHOLD_SLIDER+margin, 255):] = [255] * (256 - min(THRESHOLD_SLIDER+margin, 255))

        #     final_image = final_image.point(map, mode="L")

        inv = st.checkbox("Invert image?", False)
        if inv: final_image = ImageOps.invert(final_image)

        st.write("Preview:")
        st.image(final_image)
        st.write("Ensure the above image is **Black text on White background**")
        if st.button("Save Crop"):
            st.session_state.img = final_image
            st.rerun()

    # MARK: | Upload and Character Input
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                font_image = st.file_uploader(
                    "📤 Upload Image",
                    type=["jpg", "png", "jpeg"],
                    accept_multiple_files=False,
                )

                if font_image:
                    if "img" not in st.session_state:
                        st.session_state.img = None
                        img = Image.open(font_image)
                        crop(img)
                    else:
                        st.image(
                            st.session_state.img,
                        )
                        if st.button("Change Image"):
                            st.session_state.pop("img")
                            img = Image.open(font_image)
                            crop(img)
                else:
                    if "img" in st.session_state:
                        st.session_state.pop("img")
                    st.error("⚠️ Please upload an image")

        with col2:
            character = st.text_input(
                "🅱️ Enter Character",
                max_chars=1,
                placeholder="B",
            )

        predict_button = st.button(
            "Predict",
            use_container_width=True,
            disabled="img" not in st.session_state or not character,
        )

    # MARK: | Prediction and output
    if predict_button:
        if "model" not in st.session_state:
            st.error("⚠️ The model is not loaded yet, please try again.")

        else:
            st.divider()

            st.header("📊 Output")

            output = [
                {"Font": font_name, "Confidence": score}
                for font_name, score in st.session_state.model.predict(
                    st.session_state.img, ord(character)
                )
            ]

            prediction = pd.DataFrame(output)

            st.dataframe(
                prediction,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(width="large")
                },
            )

            top_font = output[0]["Font"]
            st.success(f"Top font: {top_font}")

st.divider()

# MARK: Info
st.header("📈 About the Dataset")
st.write(
    """
The dataset used to train the model is synthetically generated with the help of the **[Pillow](https://pillow.readthedocs.io/en/stable/)** library. It is created using 167 commonly available fonts found on most modern Windows devices.

For each font, images are generated for the following characters:
- **Uppercase letters**: `A-Z`
- **Lowercase letters**: `a-z`
- **Digits**: `0-9`

Each image is preprocessed to a consistent size of **64x64 pixels** and represented as a dataset for training.
""")

with st.expander("Fonts used in the dataset", icon="📝"):
    df = pd.read_csv("frontend_demo.csv")
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Font Name": st.column_config.Column(
                width="small", pinned=True
            ),
            "Sample Text": st.column_config.ImageColumn(
                label=None, width="large", help=None
            ),
        },
        use_container_width=True,
    )

st.header("🤖 About the Model")
st.write(
    """
The model used in this app is a **Convolutional Neural Network (CNN)** trained on the font dataset. CNNs are a type of deep learning models that are particularly effective for image classification tasks.
    """
)

# with st.expander("Confusion Matrix", icon="⁉️"):
#     st.image(
#         "assets/Confusion Matrix (P).png",
#         use_container_width=True,
#         caption='Confusion Matrix of the Character "P"',
#     )

st.write(
    """
## 💻 Developers

- 2022A7PS0032U - **Ryan Abraham Philip**
- 2022A7PS0034U - **Sreenikethan Iyer**
- 2022A7PS0140U - **Akamksha Ranil**
- 2022A7PS0126U - **Adeeb Husain**
- 2022A7PS0050U - **Aditya Agarwal**
    """
)

# Load the model after the webpage has been rendered
if "model" not in st.session_state:
    st.session_state["model"] = SimbleModel(MODEL_PATH, ALPNAMES_PATH)
    st.rerun()
