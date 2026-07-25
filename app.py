import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

from utils import (
    preprocess_image,
    make_gradcam_heatmap,
    overlay_heatmap,
    CLASSES
)



# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="IDC Classifier",
    page_icon="🔬",
    layout="centered"
)



# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "IDC_MobileNetV2_model.keras"
    )

    return model



model = load_model()



# -----------------------------
# Interface
# -----------------------------

st.title(
    "Breast Cncer Image Classifier"
)


st.write(
    """
Upload a breast histopathology image.
The model predicts IDC class and provides
Grad-CAM visual explanation.
"""
)



uploaded_file = st.file_uploader(
    "Upload image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)



if uploaded_file:


    image = Image.open(
        uploaded_file
    )


    image = image.convert(
        "RGB"
    )


    st.subheader(
        "Uploaded Image"
    )


    st.image(
        image,
        width=350
    )



    # -----------------------------
    # Preprocess
    # -----------------------------

    processed = preprocess_image(
        image
    )



    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(
        processed,
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(prediction)
    )


    predicted_class = CLASSES[
        predicted_index
    ]


    confidence = float(
        np.max(prediction)
    )



    st.subheader(
        "Prediction Result"
    )


    st.success(
        f"Prediction: {predicted_class}"
    )


    st.info(
        f"Confidence: {confidence:.2%}"
    )



    # -----------------------------
    # Probabilities
    # -----------------------------

    st.subheader(
        "Class Probabilities"
    )


    for i, probability in enumerate(prediction):

        st.write(
            f"{CLASSES[i]} : {probability:.2%}"
        )



    # -----------------------------
    # Grad-CAM
    # -----------------------------

    st.subheader(
        "Model Explanation (Grad-CAM)"
    )


    try:

        heatmap = make_gradcam_heatmap(
            processed,
            model,
            "Conv_1"
        )


        original = np.array(
            image
        )


        explanation = overlay_heatmap(
            original,
            heatmap
        )


        st.image(
            explanation,
            caption=
            "Highlighted regions influencing model prediction",
            width=400
        )


    except Exception as e:

        st.warning(
            f"Grad-CAM generation failed: {e}"
        )