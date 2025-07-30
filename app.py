import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
import glob
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Lichen Air Quality Indicator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Background Image Function ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# --- Custom CSS for Styling ---
def get_custom_css(background_file):
    bg_image_base64 = get_base64_of_bin_file(background_file) if background_file else None
    if bg_image_base64:
        background_style = f"""
            background-image: url("data:image/png;base64,{bg_image_base64}");
        """
    else:
        st.warning("`background.png` not found. Using a default dark background.")
        background_style = "background-color: #1a1a1a;"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Roboto:wght@400;700&display=swap');
        .stApp {{
            {background_style}
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        header[data-testid="stHeader"] {{
            display: none;
        }}
        .main .block-container {{
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 20px;
            padding: 2rem 5rem;
        }}
        .main-title {{
            font-family: 'Montserrat', sans-serif;
            font-size: 3.5rem;
            font-weight: 700;
            color: #FFFFFF;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
            text-align: center;
            padding-bottom: 0.5rem;
        }}
        .subtitle {{
             font-family: 'Roboto', sans-serif;
             text-align: center;
             font-size: 1.2rem;
             color: #E0E0E0;
        }}
        .stMarkdown, p, .stFileUploader label, li {{
            font-family: 'Roboto', sans-serif;
            color: #E0E0E0;
            font-size: 1.1rem;
            line-height: 1.6;
        }}
        strong {{
            color: #4CAF50;
        }}
        .result-card {{
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 15px;
            padding: 25px;
            margin-top: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .result-card h2 {{
             font-family: 'Montserrat', sans-serif;
             color: #FFFFFF;
             margin-top: 0;
        }}
    </style>
    """

st.markdown(get_custom_css("background.png"), unsafe_allow_html=True)

# --- Lichen Information Map ---
LICHEN_DATA = {
    'Usnea_filipendula': {...},  # same as before
    # populate with all species data
}

# --- Model Loading (Using Keras) ---
@st.cache_resource
def load_models_and_labels():
    models = []
    # find all .keras model files
    model_files = sorted(glob.glob("*.keras"))
    for mfile in model_files:
        try:
            models.append(tf.keras.models.load_model(mfile))
        except Exception as e:
            st.error(f"Error loading {mfile}: {e}")
    # load labels
    try:
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("labels.txt not found. Please add a labels.txt file.")
        labels = []
    return models, labels

models, labels = load_models_and_labels()

# --- Prediction Function (Using Keras Models) ---
def predict(image_data):
    if not models or not labels:
        return None
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # normalize
    img_expanded = np.expand_dims(img_array, axis=0)

    preds = [model.predict(img_expanded) for model in models]
    avg_preds = np.mean(preds, axis=0)
    predicted_index = np.argmax(avg_preds)
    predicted_label = labels[predicted_index]
    formatted_label = predicted_label.replace(" ", "_")
    return LICHEN_DATA.get(formatted_label, None)

# --- Main Application UI ---
st.markdown('<h1 class="main-title">Lichen Air Quality Indicator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a photo of a lichen to assess the local air quality based on its species.</p>', unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose an image file (or drag and drop)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_data = uploaded_file.getvalue()
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(image_data, caption="Your Uploaded Lichen", use_container_width=True)

    with col2:
        with st.spinner("Analyzing lichen species..."):
            prediction_info = predict(image_data)
        if prediction_info:
            result_html = f"""
            <div class="result-card">
                <h2>{prediction_info['common_name']}</h2>
                <p><strong>Tolerance to Air Pollution:</strong> {prediction_info['tolerance']}</p>
                <p><strong>Air Quality (AQI) Indicated:</strong> {prediction_info['aqi_category']} (Estimated Range: {prediction_info['aqi_range']})</p>
                <p><strong>What This Lichen's Presence Suggests:</strong> {prediction_info['inference']}</p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
        else:
            st.error("Could not analyze the image. Please try another one.")
else:
    st.info("Awaiting image upload to begin analysis.")
