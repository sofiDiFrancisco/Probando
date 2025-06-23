import streamlit as st
import torch
from PIL import Image
import os
from utils import load_model, preprocess_image

# Constants
MODEL_PATH = "modelo.pth"  # Replace with the actual path if different
NUM_CLASSES = 6
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
try:
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {MODEL_PATH}. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Class names
class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

# Streamlit app
st.title("Fruit Classifier")
st.header("Upload an image of a fruit to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image, IMG_SIZE, DEVICE)

        # Perform inference
        with torch.no_grad():
            outputs = model(preprocessed_image)
            _, predicted_class_index = torch.max(outputs, 1)

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index.item()]

        # Display the prediction
        st.write(f"Prediction: {predicted_class_name}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
