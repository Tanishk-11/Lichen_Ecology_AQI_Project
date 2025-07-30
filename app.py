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
        background_style = f"background-image: url(\"data:image/png;base64,{bg_image_base64}\");"
    else:
        background_style = "background-color: #1a1a1a;"
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Roboto:wght@400;700&display=swap');
        .stApp {{ {background_style} background-size: cover; background-repeat: no-repeat; background-attachment: fixed; }}
        header[data-testid=\"stHeader\"] {{ display: none; }}
        .main .block-container {{ background-color: rgba(0,0,0,0.6); border-radius: 20px; padding: 2rem 5rem; }}
        .main-title {{ font-family: 'Montserrat', sans-serif; font-size: 3.5rem; font-weight: 700; color: #FFF; text-shadow: 3px 3px 6px rgba(0,0,0,0.7); text-align: center; padding-bottom: .5rem; }}
        .subtitle {{ font-family: 'Roboto', sans-serif; text-align: center; font-size: 1.2rem; color: #E0E0E0; }}
        .stMarkdown, p, .stFileUploader label, li {{ font-family: 'Roboto', sans-serif; color: #E0E0E0; font-size: 1.1rem; line-height: 1.6; }}
        strong {{ color: #4CAF50; }}
        .result-card {{ background-color: rgba(0,0,0,0.7); border-radius: 15px; padding: 25px; margin-top: 2rem; border: 1px solid rgba(255,255,255,0.1); }}
        .result-card h2 {{ font-family: 'Montserrat', sans-serif; color: #FFF; margin-top: 0; }}
    </style>
    """

st.markdown(get_custom_css("background.png"), unsafe_allow_html=True)

# --- Lichen Information Map (inline) ---
LICHEN_DATA = {
    'Usnea_filipendula': {
        'common_name': "Old Man's Beard",
        'tolerance': "Very Sensitive",
        'aqi_category': "Good",
        'aqi_range': "0-50",
        'inference': "Air is very clean."
    },
    'Xanthoria_parietina': {
        'common_name': "Common Orange Lichen",
        'tolerance': "Moderately Sensitive",
        'aqi_category': "Moderate",
        'aqi_range': "51-100",
        'inference': "Air quality is acceptable."
    },
    'Parmelia_saxatilis': {
        'common_name': "Shield Lichen",
        'tolerance': "Moderately Tolerant",
        'aqi_category': "Unhealthy for Sensitive Groups",
        'aqi_range': "101-150",
        'inference': "Sensitive individuals should reduce prolonged outdoor exertion."
    },
    'Lecanora_conizaeoides': {
        'common_name': "Smoky Eye Lichen",
        'tolerance': "Tolerant",
        'aqi_category': "Unhealthy",
        'aqi_range': "151-200",
        'inference': "Everyone may begin to experience health effects."
    },
    'Phaeophyscia_orbicularis': {
        'common_name': "Orange Cobblestone Lichen",
        'tolerance': "Very Tolerant",
        'aqi_category': "Very Unhealthy",
        'aqi_range': "201-300",
        'inference': "Health warnings of emergency conditions."
    },
    # Add more species entries as needed
}

# --- Preprocessing ---
def preprocess_image(image_bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# --- Model Loading ---
@st.cache_resource
def load_models_and_labels():
    models = []
    for path in sorted(glob.glob("*.keras")) + sorted(glob.glob("*.h5")):
        try:
            models.append(tf.keras.models.load_model(path, compile=False))
        except Exception as e:
            st.error(f"Couldn’t load '{path}': {e}")
    try:
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f]
    except FileNotFoundError:
        st.error("labels.txt not found. Please add a labels.txt file with one label per line.")
        labels = []
    return models, labels

models, labels = load_models_and_labels()

# --- Prediction Function ---
def predict(image_bytes):
    if not models or not labels:
        return None
    x = preprocess_image(image_bytes)
    preds = [model.predict(x) for model in models]
    avg = np.mean(preds, axis=0)
    idx = int(np.argmax(avg))
    label = labels[idx]
    key = label.replace(' ', '_')
    return LICHEN_DATA.get(key)

# --- Main App ---
st.markdown('<h1 class="main-title">Lichen Air Quality Indicator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a photo of a lichen to assess local air quality based on its species.</p>', unsafe_allow_html=True)
st.markdown('---')

uploaded = st.file_uploader("Choose an image file", type=['jpg','jpeg','png'])
if uploaded:
    img_bytes = uploaded.getvalue()
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.image(img_bytes, caption="Uploaded Lichen", use_container_width=True)
    with col2:
        with st.spinner("Analyzing..."):
            info = predict(img_bytes)
        if info:
            st.markdown(f"""
            <div class="result-card">
                <h2>{info['common_name']}</h2>
                <p><strong>Tolerance to Air Pollution:</strong> {info['tolerance']}</p>
                <p><strong>Air Quality (AQI):</strong> {info['aqi_category']} ({info['aqi_range']})</p>
                <p><strong>What This Lichen Suggests:</strong> {info['inference']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Could not identify lichen. Check your data or try another image.")
else:
    st.info("Awaiting image upload...")
