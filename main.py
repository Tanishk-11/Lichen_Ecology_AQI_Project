# # main.py
# import io
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from fastapi import FastAPI, File, UploadFile, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from PIL import Image

# # Initialize the FastAPI app
# app = FastAPI()

# # Mount static files directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Initialize Jinja2 templates
# templates = Jinja2Templates(directory="templates")

# # --- Model and Label Loading ---
# # Load both models for the ensemble
# try:
#     # With matching TF versions, compile=False is not needed
#     model1 = tf.keras.models.load_model('true_model_version_1.keras')
#     # Load the MobileNetV2 model with the custom preprocess_input object
#     model2 = tf.keras.models.load_model(
#         'true_mobilenetv2_lichen_model_1.keras',
#         custom_objects={'preprocess_input': preprocess_input}
#     )
#     with open("labels.txt", "r") as f:
#         labels = [line.strip() for line in f.readlines()]
# except Exception as e:
#     print(f"Error loading models or labels: {e}")
#     model1 = None
#     model2 = None
#     labels = []

# # Lichen data for providing detailed information
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
#         'inference': "Finding this lichen indicates elevated nitrogen levels in the air, a form of pollution often caused by agricultural fertilizers or vehicle emissions.",
#         'aqi_category': 'Unhealthy for Sensitive Groups'
#     },
#     'Xanthoria_parietina': {
#         'common_name': 'Common Orange Lichen', 'tolerance': 'Very Tolerant', 'aqi_range': '125 - 200+',
#         'inference': "This lichen is a sign of high nitrogen and nutrient pollution. It thrives in areas with significant agricultural runoff or heavy traffic, indicating poor air quality.",
#         'aqi_category': 'Unhealthy'
#     },
#     'Lecanora_conizaeoides': {
#         'common_name': 'Powdery-crust Lichen', 'tolerance': 'Very Tolerant', 'aqi_range': '150 - 250+',
#         'inference': "The presence of this lichen indicates high levels of sulfur dioxide (SO2) and acid rain. It is often one of the few lichens found in heavily industrialized or urban areas with poor air quality.",
#         'aqi_category': 'Very Unhealthy'
#     }
# }

# # Function to process the image and make predictions using the ensemble
# def predict(uploaded_file: UploadFile):
#     try:
#         # Pass the file-like object from the UploadFile directly to Pillow
#         image = Image.open(uploaded_file.file).convert("RGB")
#         image = image.resize((224, 224))
#         image_array = np.array(image)
#         image_array = np.expand_dims(image_array, axis=0)

#         if model1 and model2:
#             # Get predictions from both models.
#             # model2 will handle its own preprocessing since it's part of the loaded model.
#             predictions1 = model1.predict(image_array)
#             predictions2 = model2.predict(image_array)
            
#             # Average the predictions for the ensemble
#             avg_predictions = (predictions1 + predictions2) / 2
            
#             # Get the final predicted class
#             predicted_class = labels[np.argmax(avg_predictions)]
#             return LICHEN_DATA.get(predicted_class.replace(' ', '_'))
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#     return None

# # Route for the main page
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # Route for handling file uploads and predictions
# @app.post("/predict/", response_class=HTMLResponse)
# async def predict_image(request: Request, file: UploadFile = File(...)):
#     if not file.filename:
#         return templates.TemplateResponse("index.html", {"request": request, "prediction": None})
    
#     # Pass the UploadFile object directly to the predict function
#     prediction_info = predict(file)
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction_info})









# main.py
import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Initialize the FastAPI app
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# --- Model and Label Loading ---
# Load both models for the ensemble
try:
    # With matching TF versions, compile=False is not needed
    model1 = tf.keras.models.load_model('true_model_version_1.keras')
    # Load the MobileNetV2 model with the custom preprocess_input object
    model2 = tf.keras.models.load_model(
        'true_mobilenetv2_lichen_model_1.keras',
        custom_objects={'preprocess_input': preprocess_input}
    )
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Error loading models or labels: {e}")
    model1 = None
    model2 = None
    labels = []

# Lichen data for providing detailed information
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
        'inference': "Finding this lichen indicates elevated nitrogen levels in the air, a form of pollution often caused by agricultural fertilizers or vehicle emissions.",
        'aqi_category': 'Unhealthy for Sensitive Groups'
    },
    'Xanthoria_parietina': {
        'common_name': 'Common Orange Lichen', 'tolerance': 'Very Tolerant', 'aqi_range': '125 - 200+',
        'inference': "This lichen is a sign of high nitrogen and nutrient pollution. It thrives in areas with significant agricultural runoff or heavy traffic, indicating poor air quality.",
        'aqi_category': 'Unhealthy'
    },
    'Lecanora_conizaeoides': {
        'common_name': 'Powdery-crust Lichen', 'tolerance': 'Very Tolerant', 'aqi_range': '150 - 250+',
        'inference': "The presence of this lichen indicates high levels of sulfur dioxide (SO2) and acid rain. It is often one of the few lichens found in heavily industrialized or urban areas with poor air quality.",
        'aqi_category': 'Very Unhealthy'
    }
}

# Function to process the image and make predictions using the ensemble
def predict(image_data: bytes):
    try:
        # Use io.BytesIO to create a file-like object from the image bytes
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        if model1 and model2:
            # Get predictions from both models.
            predictions1 = model1.predict(image_array)
            predictions2 = model2.predict(image_array)
            
            # Average the predictions for the ensemble
            avg_predictions = (predictions1 + predictions2) / 2
            
            # Get the final predicted class
            predicted_class = labels[np.argmax(avg_predictions)]
            return LICHEN_DATA.get(predicted_class.replace(' ', '_'))
    except Exception as e:
        print(f"Error during prediction: {e}")
    return None

# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for handling file uploads and predictions
@app.post("/predict/", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        return templates.TemplateResponse("index.html", {"request": request})
    
    # Read the image data once
    image_data = await file.read()
    
    # Get prediction from the bytes
    prediction_info = predict(image_data)
    
    # Encode the same bytes for display on the webpage
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "prediction": prediction_info, 
            "image_base64": image_base64
        }
    )
