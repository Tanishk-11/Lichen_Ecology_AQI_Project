import tensorflow as tf
import os

print(f"Using TensorFlow version: {tf.__version__}")

# --- File paths ---
original_effnet_path = 'true_model_version_1.keras'
original_mobilenet_path = 'true_mobilenetv2_lichen_model_1.keras'

updated_effnet_path = 'updated_efficientnet.keras'
updated_mobilenet_path = 'updated_mobilenet.keras'

# --- Update EfficientNet Model ---
if os.path.exists(original_effnet_path):
    print(f"\nLoading original model: {original_effnet_path}")
    model_effnet = tf.keras.models.load_model(original_effnet_path)
    print("Re-saving model in the latest format...")
    model_effnet.save(updated_effnet_path)
    print(f"Successfully saved updated model to: {updated_effnet_path}")
else:
    print(f"ERROR: Could not find {original_effnet_path}")

# --- Update MobileNetV2 Model ---
if os.path.exists(original_mobilenet_path):
    print(f"\nLoading original model: {original_mobilenet_path}")
    custom_objects = {"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input}
    model_mobilenet = tf.keras.models.load_model(original_mobilenet_path, custom_objects=custom_objects)
    print("Re-saving model in the latest format...")
    model_mobilenet.save(updated_mobilenet_path)
    print(f"Successfully saved updated model to: {updated_mobilenet_path}")
else:
    print(f"ERROR: Could not find {original_mobilenet_path}")

print("\nModel update process finished.")