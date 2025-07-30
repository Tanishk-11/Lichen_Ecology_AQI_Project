import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
import glob

# Lower TF log level
tf.get_logger().setLevel('ERROR')

# --- Page Configuration ---
st.set_page_config(page_title="Lichen Air Quality Indicator", layout="wide")

# --- Background Styling ---
@st.cache_data
def get_base64(bin_file):
    try:
        return base64.b64encode(open(bin_file, 'rb').read()).decode()
    except:
        return None

def inject_css(bg_file):
    b64 = get_base64(bg_file)
    bg = f"url('data:image/png;base64,{b64}')" if b64 else '#1a1a1a'
    st.markdown(f"""
    <style>
      .stApp {{ background: {bg} center/cover no-repeat fixed; }}
      header, footer {{ visibility: hidden; }}
      .block-container {{ background: rgba(0,0,0,0.6); border-radius: 20px; padding: 2rem; }}
      .main-title {{ font-family: Montserrat; font-size: 3rem; color: #fff; text-align: center; }}
      .subtitle {{ font-family: Roboto; color: #ddd; text-align: center; }}
      .result-card {{ background: rgba(0,0,0,0.7); border-radius: 15px; padding: 20px; margin-top: 1rem; }}
      .result-card h2 {{ color: #4caf50; }}
    </style>
    """, unsafe_allow_html=True)

inject_css('background.png')

# --- Lichen Data Inline ---
LICHEN_DATA = {
    'Usnea_filipendula': {'common_name': "Old Man's Beard", 'tolerance': "Very Sensitive", 'aqi_category': "Good", 'aqi_range': "0-50", 'inference': "Air is very clean."},
    'Xanthoria_parietina': {'common_name': "Common Orange Lichen", 'tolerance': "Moderately Sensitive", 'aqi_category': "Moderate", 'aqi_range': "51-100", 'inference': "Air quality is acceptable."},
    'Parmelia_saxatilis': {'common_name': "Shield Lichen", 'tolerance': "Moderately Tolerant", 'aqi_category': "Unhealthy for Sensitive Groups", 'aqi_range': "101-150", 'inference': "Sensitive individuals should reduce prolonged outdoor exertion."},
    'Lecanora_conizaeoides': {'common_name': "Smoky Eye Lichen", 'tolerance': "Tolerant", 'aqi_category': "Unhealthy", 'aqi_range': "151-200", 'inference': "Everyone may begin to experience health effects."},
    'Phaeophyscia_orbicularis': {'common_name': "Orange Cobblestone Lichen", 'tolerance': "Very Tolerant", 'aqi_category': "Very Unhealthy", 'aqi_range': "201-300", 'inference': "Health warnings of emergency conditions."}
}

# --- Image Preprocessing ---
def preprocess(img_bytes, size=(224,224)):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(size)
    return np.expand_dims(np.array(img) / 255.0, 0)

# --- Load Models & Labels ---
@st.cache_resource
def load_models_and_labels():
    models, labels = [], []
    custom_objects = {
        'RandomFlip': tf.keras.layers.RandomFlip,
        'RandomRotation': tf.keras.layers.RandomRotation,
        'RandomZoom': tf.keras.layers.RandomZoom,
        'RandomContrast': tf.keras.layers.RandomContrast
    }
    # load all Keras files
    files = sorted(glob.glob('*.keras')) + sorted(glob.glob('*.h5'))
    st.info(f"Found model files: {files}")
    for path in files:
        try:
            m = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
            models.append(m)
        except Exception as e:
            st.error(f"Error loading '{os.path.basename(path)}': {e}")
    # load labels
    try:
        labels = [l.strip() for l in open('labels.txt')]
    except Exception:
        st.error('labels.txt missing or unreadable.')
    return models, labels

models, labels = load_models_and_labels()

# --- Prediction Function ---
def predict(img_bytes):
    if not models or not labels:
        return None
    x = preprocess(img_bytes)
    preds = []
    for m in models:
        # ensure channel match
        expected = m.input_shape[-1]
        if x.shape[-1] != expected:
            if expected == 1:
                x_use = np.mean(x, axis=-1, keepdims=True)
            else:
                x_use = np.repeat(x, 3, axis=-1)
        else:
            x_use = x
        preds.append(m.predict(x_use))
    avg = np.mean(preds, axis=0)
    idx = int(np.argmax(avg))
    key = labels[idx].replace(' ', '_')
    return LICHEN_DATA.get(key)

# --- UI ---
st.markdown('<h1 class="main-title">Lichen Air Quality Indicator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a lichen photo to assess air quality.</p>', unsafe_allow_html=True)
st.markdown('---')

file = st.file_uploader('', type=['jpg', 'png', 'jpeg'])
if file:
    img_bytes = file.getvalue()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img_bytes, use_container_width=True)
    with c2:
        info = predict(img_bytes)
        if info:
            st.markdown(f"""
            <div class="result-card">
              <h2>{info['common_name']}</h2>
              <p><strong>Tolerance:</strong> {info['tolerance']}</p>
              <p><strong>AQI:</strong> {info['aqi_category']} ({info['aqi_range']})</p>
              <p><strong>Inference:</strong> {info['inference']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error('Lichen not identified.')
else:
    st.info('Awaiting upload...')