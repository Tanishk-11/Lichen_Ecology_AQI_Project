# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io
# import base64

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Lichen Air Quality Indicator",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # --- Background Image Function ---
# @st.cache_data
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # --- Custom CSS for Styling ---
# def get_custom_css(background_file):
#     try:
#         bg_image_base64 = get_base64_of_bin_file(background_file)
#         background_style = f"""
#             background-image: url("data:image/png;base64,{bg_image_base64}");
#         """
#     except FileNotFoundError:
#         st.warning("`background.png` not found. Using a default dark background.")
#         background_style = "background-color: #1a1a1a;"

#     return f"""
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Roboto:wght@400;700&display=swap');
#         .stApp {{
#             {background_style}
#             background-size: cover;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}
#         header[data-testid="stHeader"] {{
#             display: none;
#         }}
#         .main .block-container {{
#             background-color: rgba(0, 0, 0, 0.6);
#             border-radius: 20px;
#             padding: 2rem 5rem;
#         }}
#         .main-title {{
#             font-family: 'Montserrat', sans-serif;
#             font-size: 3.5rem;
#             font-weight: 700;
#             color: #FFFFFF;
#             text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
#             text-align: center;
#             padding-bottom: 0.5rem;
#         }}
#         .subtitle {{
#              font-family: 'Roboto', sans-serif;
#              text-align: center;
#              font-size: 1.2rem;
#              color: #E0E0E0;
#         }}
#         .stMarkdown, p, .stFileUploader label, li {{
#             font-family: 'Roboto', sans-serif;
#             color: #E0E0E0;
#             font-size: 1.1rem;
#             line-height: 1.6;
#         }}
#         strong {{
#             color: #4CAF50;
#         }}
#         .result-card {{
#             background-color: rgba(0, 0, 0, 0.7);
#             border-radius: 15px;
#             padding: 25px;
#             margin-top: 2rem;
#             border: 1px solid rgba(255, 255, 255, 0.1);
#         }}
#         .result-card h2 {{
#              font-family: 'Montserrat', sans-serif;
#              color: #FFFFFF;
#              margin-top: 0;
#         }}
#     </style>
#     """

# # Apply the custom CSS
# st.markdown(get_custom_css("background.png"), unsafe_allow_html=True)


# # --- Lichen Information Map (UPDATED) ---
# # Rephrased the 'inference' text to be a direct conclusion about the environment.
# LICHEN_DATA = {
#     'Usnea_filipendula': {
#         'common_name': 'Fishbone Beard Lichen', 'tolerance': 'Very Sensitive', 'aqi_range': '0 - 50',
#         'inference': "Finding this lichen is a strong sign that the air in this location is consistently clean and unpolluted, with very low levels of sulfur dioxide (SO2) and other pollutants.",
#         'aqi_category': 'Good'
#     },
#     'Ramalina_farinacea': {
#         'common_name': 'Cartilage Lichen', 'tolerance': 'Sensitive', 'aqi_range': '0 - 50',
#         'inference': "This lichen's presence indicates very clean air. It is highly sensitive to pollutants like sulfur dioxide (SO2) and acid rain, so it only thrives in healthy environments.",
#         'aqi_category': 'Good'
#     },
#     'Evernia_prunastri': {
#         'common_name': 'Oakmoss', 'tolerance': 'Sensitive', 'aqi_range': '25 - 75',
#         'inference': "This lichen's presence suggests the air quality is good. It is a reliable indicator of low sulfur dioxide (SO2), though it can tolerate very minor levels of nitrogen pollution.",
#         'aqi_category': 'Good to Moderate'
#     },
#     'Hypogymnia_physodes': {
#         'common_name': "Monk's-hood Lichen", 'tolerance': 'Intermediate', 'aqi_range': '50 - 100',
#         'inference': "Finding this common lichen suggests the air quality is moderate. It can tolerate some levels of SO2 and acidic pollution, typical of suburban areas or locations with light traffic.",
#         'aqi_category': 'Moderate'
#     },
#     'Parmelia_sulcata': {
#         'common_name': 'Shield Lichen', 'tolerance': 'Intermediate', 'aqi_range': '50 - 125',
#         'inference': "This adaptable lichen indicates moderate air quality. Its ability to survive in moderately polluted environments means the air is not pristine, but likely not heavily polluted either.",
#         'aqi_category': 'Moderate'
#     },
#     'Flavoparmelia_caperata': {
#         'common_name': 'Common Greenshield', 'tolerance': 'Intermediate', 'aqi_range': '50 - 125',
#         'inference': "The presence of this lichen suggests a moderate level of air quality. It can handle general pollution but will disappear if conditions worsen significantly.",
#         'aqi_category': 'Moderate'
#     },
#     'Punctelia_rudecta': {
#         'common_name': 'Rough Speckled Shield', 'tolerance': 'Tolerant', 'aqi_range': '100 - 175',
#         'inference': "This lichen's presence suggests moderate air pollution, specifically with higher levels of nitrogen (nutrient enrichment), often from nearby agriculture or traffic.",
#         'aqi_category': 'Unhealthy for Sensitive Groups'
#     },
#     'Physcia_stellaris': {
#         'common_name': 'Star Rosette Lichen', 'tolerance': 'Tolerant', 'aqi_range': '100 - 175',
#         'inference': "Finding this lichen indicates elevated nitrogen levels in the air, a form of pollution often caused by agricultural fertilizers or vehicle exhaust.",
#         'aqi_category': 'Unhealthy for Sensitive Groups'
#     },
#     'Xanthoria_parietina': {
#         'common_name': 'Common Orange Lichen', 'tolerance': 'Very Tolerant', 'aqi_range': '125 - 200',
#         'inference': "This lichen is a strong indicator that the air in this location has high nitrogen levels. It thrives in polluted areas with excess nutrients from traffic or agriculture.",
#         'aqi_category': 'Unhealthy'
#     },
#     'Lecanora_conizaeoides': {
#         'common_name': 'Powdery Bark Lichen', 'tolerance': 'Extremely Tolerant', 'aqi_range': '175 - 250',
#         'inference': "This lichen is a sign of significant air pollution. Its extreme tolerance to sulfur dioxide (SO2) and acid rain allows it to survive in environments where most other lichens cannot.",
#         'aqi_category': 'Unhealthy to Very Unhealthy'
#     }
# }


# # --- Model Loading ---
# @st.cache_resource
# def load_models():
#     """Loads and returns the full Keras models and labels."""
#     try:
#         effnet_model = tf.keras.models.load_model("true_model_version_1.keras")
#         custom_objects = {"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input}
#         mobilenet_model = tf.keras.models.load_model(
#             "true_mobilenetv2_lichen_model_1.keras",
#             custom_objects=custom_objects
#         )
#         with open("labels.txt", "r") as f:
#             labels = [line.strip() for line in f.readlines()]
#         return effnet_model, mobilenet_model, labels
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         st.error("Please make sure 'true_model_version_1.keras', 'true_mobilenetv2_lichen_model_1.keras', and 'labels.txt' are in the same folder.")
#         return None, None, None

# effnet_model, mobilenet_model, labels = load_models()


# # --- Prediction Function ---
# def predict(image_data):
#     if effnet_model is None or mobilenet_model is None:
#         return None
#     img = Image.open(io.BytesIO(image_data)).convert('RGB')
#     img_resized = img.resize((224, 224))
#     img_array = np.array(img_resized, dtype=np.float32)
#     img_expanded = np.expand_dims(img_array, axis=0)
#     effnet_input = img_expanded
#     mobilenet_input = img_expanded
#     effnet_preds = effnet_model.predict(effnet_input)
#     mobilenet_preds = mobilenet_model.predict(mobilenet_input)
#     ensemble_preds = (effnet_preds + mobilenet_preds) / 2.0
#     predicted_index = np.argmax(ensemble_preds)
#     predicted_label_from_file = labels[predicted_index]
#     formatted_label = predicted_label_from_file.replace(" ", "_")
#     predicted_info = LICHEN_DATA.get(formatted_label, None)
#     return predicted_info


# # --- Main Application UI ---
# st.markdown('<h1 class="main-title">Lichen Air Quality Indicator</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">Upload a photo of a lichen to assess the local air quality based on its species.</p>', unsafe_allow_html=True)
# st.markdown("---")

# uploaded_file = st.file_uploader(
#     "Choose an image file (or drag and drop)",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:
#     image_data = uploaded_file.getvalue()
#     col1, col2 = st.columns([1, 1.5])

#     with col1:
#         st.image(image_data, caption="Your Uploaded Lichen", use_container_width=True)

#     with col2:
#         with st.spinner("Analyzing lichen species..."):
#             prediction_info = predict(image_data)
        
#         if prediction_info:
#             # FIX: Rewrote the result block with clearer, more user-friendly labels.
#             result_html = f"""
#             <div class="result-card">
#                 <h2>{prediction_info['common_name']}</h2>
#                 <p><strong>Tolerance to Air Pollution:</strong> {prediction_info['tolerance']}</p>
#                 <p><strong>Air Quality (AQI) Indicated:</strong> {prediction_info['aqi_category']} (Estimated Range: {prediction_info['aqi_range']})</p>
#                 <p><strong>What This Lichen's Presence Suggests:</strong> {prediction_info['inference']}</p>
#             </div>
#             """
#             st.markdown(result_html, unsafe_allow_html=True)
#         else:
#             st.error("Could not analyze the image. Please try another one.")
# else:
#     st.info("Awaiting image upload to begin analysis.")










import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Lichen Air Quality Indicator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Background Image Function ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Custom CSS for Styling ---
def get_custom_css(background_file):
    try:
        bg_image_base64 = get_base64_of_bin_file(background_file)
        background_style = f"""
            background-image: url("data:image/png;base64,{bg_image_base64}");
        """
    except FileNotFoundError:
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
    'Usnea_filipendula': {
        'common_name': 'Fishbone Beard Lichen', 'tolerance': 'Very Sensitive', 'aqi_range': '0 - 50',
        'inference': "Finding this lichen is a strong sign that the air in this location is consistently clean and unpolluted, with very low levels of sulfur dioxide (SO2) and other pollutants.",
        'aqi_category': 'Good'
    },
    'Ramalina_farinacea': {
        'common_name': 'Cartilage Lichen', 'tolerance': 'Sensitive', 'aqi_range': '0 - 50',
        'inference': "This lichen's presence indicates very clean air. It is highly sensitive to pollutants like sulfur dioxide (SO2) and acid rain, so it only thrives in healthy environments.",
        'aqi_category': 'Good'
    },
    'Evernia_prunastri': {
        'common_name': 'Oakmoss', 'tolerance': 'Sensitive', 'aqi_range': '25 - 75',
        'inference': "This lichen's presence suggests the air quality is good. It is a reliable indicator of low sulfur dioxide (SO2), though it can tolerate very minor levels of nitrogen pollution.",
        'aqi_category': 'Good to Moderate'
    },
    'Hypogymnia_physodes': {
        'common_name': "Monk's-hood Lichen", 'tolerance': 'Intermediate', 'aqi_range': '50 - 100',
        'inference': "Finding this common lichen suggests the air quality is moderate. It can tolerate some levels of SO2 and acidic pollution, typical of suburban areas or locations with light traffic.",
        'aqi_category': 'Moderate'
    },
    'Parmelia_sulcata': {
        'common_name': 'Shield Lichen', 'tolerance': 'Intermediate', 'aqi_range': '50 - 125',
        'inference': "This adaptable lichen indicates moderate air quality. Its ability to survive in moderately polluted environments means the air is not pristine, but likely not heavily polluted either.",
        'aqi_category': 'Moderate'
    },
    'Flavoparmelia_caperata': {
        'common_name': 'Common Greenshield', 'tolerance': 'Intermediate', 'aqi_range': '50 - 125',
        'inference': "The presence of this lichen suggests a moderate level of air quality. It can handle general pollution but will disappear if conditions worsen significantly.",
        'aqi_category': 'Moderate'
    },
    'Punctelia_rudecta': {
        'common_name': 'Rough Speckled Shield', 'tolerance': 'Tolerant', 'aqi_range': '100 - 175',
        'inference': "This lichen's presence suggests moderate air pollution, specifically with higher levels of nitrogen (nutrient enrichment), often from nearby agriculture or traffic.",
        'aqi_category': 'Unhealthy for Sensitive Groups'
    },
    'Physcia_stellaris': {
        'common_name': 'Star Rosette Lichen', 'tolerance': 'Tolerant', 'aqi_range': '100 - 175',
        'inference': "Finding this lichen indicates elevated nitrogen levels in the air, a form of pollution often caused by agricultural fertilizers or vehicle exhaust.",
        'aqi_category': 'Unhealthy for Sensitive Groups'
    },
    'Xanthoria_parietina': {
        'common_name': 'Common Orange Lichen', 'tolerance': 'Very Tolerant', 'aqi_range': '125 - 200',
        'inference': "This lichen is a strong indicator that the air in this location has high nitrogen levels. It thrives in polluted areas with excess nutrients from traffic or agriculture.",
        'aqi_category': 'Unhealthy'
    },
    'Lecanora_conizaeoides': {
        'common_name': 'Powdery Bark Lichen', 'tolerance': 'Extremely Tolerant', 'aqi_range': '175 - 250',
        'inference': "This lichen is a sign of significant air pollution. Its extreme tolerance to sulfur dioxide (SO2) and acid rain allows it to survive in environments where most other lichens cannot.",
        'aqi_category': 'Unhealthy to Very Unhealthy'
    }
}

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Loads and returns the full Keras models and labels."""
    try:
        # Load EfficientNet model, compiling is not needed for inference
        effnet_model = tf.keras.models.load_model("true_model_version_1.keras", compile=False)

        # The MobileNetV2 model has a custom Lambda layer, which we handle here
        mobilenet_model = tf.keras.models.load_model(
            "true_mobilenetv2_lichen_model_1.keras",
            custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
            compile=False
        )
        
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        
        return effnet_model, mobilenet_model, labels
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please make sure model files and labels.txt are present and that your requirements.txt specifies tensorflow-cpu==2.15.0")
        return None, None, None

effnet_model, mobilenet_model, labels = load_models()

# --- Prediction Function ---
def predict(image_data):
    if effnet_model is None or mobilenet_model is None:
        return None
    
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_expanded = np.expand_dims(img_array, axis=0)
    
    # The models contain their own preprocessing layers, so we feed the raw image data
    effnet_input = img_expanded
    mobilenet_input = img_expanded
    
    effnet_preds = effnet_model.predict(effnet_input)
    mobilenet_preds = mobilenet_model.predict(mobilenet_input)
    
    ensemble_preds = (effnet_preds + mobilenet_preds) / 2.0
    predicted_index = np.argmax(ensemble_preds)
    predicted_label_from_file = labels[predicted_index]
    formatted_label = predicted_label_from_file.replace(" ", "_")
    predicted_info = LICHEN_DATA.get(formatted_label, None)
    
    return predicted_info

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