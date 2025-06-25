
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import time
import requests
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image # Import for visualization
from utils import generate_gradcam_heatmap # Import the Grad-CAM function

# --- Utility Functions (from utils.py) ---
# Define the class names (must match the order used during training)
class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

def load_model(model_path, num_classes):
    """Loads the pre-trained ResNet model."""
    model = models.resnet34(pretrained=False) # Load with pretrained=False as we load state_dict
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    try:
        # Load to CPU as Streamlit runs on CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        raise # Re-raise the exception to be caught by the calling function

    return model

def preprocess_image(image: Image.Image):
    """Preprocesses the input image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension, returns a tensor

# Note: tensor_to_image is no longer needed for this task but kept for potential future use
# def tensor_to_image(tensor):
#     """Converts a normalized tensor back to a PIL Image for display."""
#     # Inverse normalization
#     inverse_normalize = transforms.Normalize(
#         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#         std=[1/0.229, 1/0.224, 1/0.225]
#     )
#     denormalized_tensor = inverse_normalize(tensor.squeeze(0)) # Remove batch dimension

#     # Clamp values to be in [0, 1] range
#     denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)

#     # Convert tensor to PIL Image
#     image = transforms.ToPILImage()(denormalized_tensor)
#     return image


def predict_image(model, image_tensor):
    """Makes a prediction using the loaded model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0] # Get probabilities using softmax
        _, predicted_class_index = torch.max(outputs, 1)

    predicted_class_name = class_names[predicted_class_index.item()]
    # Return both the predicted class name and the probabilities
    return predicted_class_name, probabilities


def get_fruit_info_from_api(fruit_name):
    """Fetches general fruit information from the Fruityvice API."""
    try:
        # The API expects lowercase fruit names without "fresh" or "rotten"
        clean_fruit_name = fruit_name.replace('fresh', '').replace('rotten', '')

        # Handle pluralization if necessary, though the API seems to work with singular
        if clean_fruit_name.endswith('s'):
          clean_fruit_name = clean_fruit_name[:-1]

        api_url = f"https://www.fruityvice.com/api/fruit/{clean_fruit_name}"
        response = requests.get(api_url)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def get_fruit_freshness_info(predicted_class_name):
    """Provides freshness information based on the predicted class."""
    fruit_info_map = {
        'freshapples': "¡Esta manzana parece fresca y lista para comer!",
        'freshbanana': "¡Esta banana está fresca y se ve deliciosa!",
        'freshoranges': "¡Esta naranja está fresca y jugosa!",
        'rottenapples': "Esta manzana parece podrida y no debería ser consumida.",
        'rottenbanana': "Esta banana está podrida y no es adecuada para comer.",
        'rottenoranges': "Esta naranja esta podrida y debe ser desechada."
    }
    return fruit_info_map.get(predicted_class_name, "Could not retrieve freshness information for this fruit.")


# --- Streamlit App Code ---
# Configuración de la página
st.set_page_config(
    page_title="FruitScan - Freshness Detector",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .header {
        color: #2e8b57;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #3cb371;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .result-box {
        border-radius: 10px;
        padding: 1.5em;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fresh {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .rotten {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        border-radius: 10px;
        padding: 1em;
        margin: 1em 0;
    }
    .nutrition-table {
        width: 100%;
        border-collapse: collapse;
    }
    .nutrition-table th {
        background-color: #2e8b57;
        color: white;
        padding: 8px;
        text-align: left;
    }
    .nutrition-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    .nutrition-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
</style>
""", unsafe_allow_html=True)

# Header de la aplicación
st.markdown('<div class="header">🍏 FruiScan Detector de Frescura</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Sube una imagen para comprobar si tu fruta está fresca o podrida</div>', unsafe_allow_html=True)

# Barra lateral con información y configuraciones
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/415/415733.png", width=100)
    st.markdown("## About")
    st.info("""
        Esta aplicación usa un modelo ResNet34 pre-entrenado, fine-tunning en el dataset Fruits Fresh and Rotten for Classification,
        para predecir la frescura de manzanas, naranjas y bananas.
        También intenta obtener información general sobre frutas de la API Fruityvice.
        Desarrollado por el GRUPO 5.
    """)
    st.markdown("## Dataset")
    st.info("[Fruits Fresh and Rotten for Classification on Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)")
    st.markdown("## API")
    st.info("[Fruityvice API](https://www.fruityvice.com/)")
    st.markdown("---")
    st.markdown("---")
    st.write("© 2025 FruitScan")


# Main content area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')

        st.image(image, caption="Uploaded Image.", use_container_width=True)
        st.write("") # Add some space


        st.write("Classifying...")

        # Add a spinner while classifying
        with st.spinner('Analyzing image...'):
            # Load the model (ensure the path is correct relative to where app.py will run)
            # Use the correct model filename
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo.pth")
            num_classes = len(class_names) # Get the number of classes from utils
            model = load_model(model_path, num_classes)

            # Preprocess the image
            input_tensor = preprocess_image(image)

            # Make a prediction
            predicted_class_name, probabilities = predict_image(model, input_tensor)

            time.sleep(1) # Simulate processing time


        st.subheader("Prediction:")

        # Display predicted class with visual indicator
        if 'fresh' in predicted_class_name:
            st.markdown(f'<div class="result-box fresh">La fruta esta: **{predicted_class_name.replace("fresh", "").capitalize()} - FRESH** ✨</div>', unsafe_allow_html=True)
        elif 'rotten' in predicted_class_name:
             st.markdown(f'<div class="result-box rotten">La fruta esta: **{predicted_class_name.replace("rotten", "").capitalize()} - ROTTEN** 🤢</div>', unsafe_allow_html=True)
        else:
             st.info(f"The fruit is: **{predicted_class_name.capitalize()}**")


        # Get freshness information
        freshness_info = get_fruit_freshness_info(predicted_class_name)
        st.subheader("Freshness Information:")
        st.write(freshness_info)

        st.markdown("---") # Separator

        # --- Grad-CAM Visualization ---
        st.subheader("Model Attention (Grad-CAM Heatmap):")
        try:
            # For ResNet34, a common target layer is the last convolutional layer
            target_layer = model.layer4[-1]
            heatmap = generate_gradcam_heatmap(model, target_layer, input_tensor)

            # Convert the original PIL image to a numpy array and normalize for overlay
            image_np = np.array(image)
            visualization = show_cam_on_image(image_np.astype(np.float32) / 255., heatmap, use_rgb=True, alpha=0.5)

            st.image(visualization, caption="Grad-CAM Heatmap", use_container_width=True)

        except Exception as gradcam_error:
            st.warning(f"Could not generate Grad-CAM heatmap: {gradcam_error}")
            st.info("Please ensure the model architecture and target layer are correctly specified for Grad-CAM.")


        st.markdown("---") # Separator


        # Get general fruit information from API
        # Clean the predicted name for the API call
        clean_fruit_name_for_api = predicted_class_name.replace('fresh', '').replace('rotten', '')
        if clean_fruit_name_for_api.endswith('s'):
          clean_fruit_name_for_api = clean_fruit_name_for_api[:-1]

        api_info = get_fruit_info_from_api(clean_fruit_name_for_api)

        if api_info:
            st.subheader(f"General Fruit Information ({api_info.get('name', 'N/A')})")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Family:** {api_info.get('family', 'N/A')}")
            st.write(f"**Order:** {api_info.get('order', 'N/A')}")
            st.write(f"**Genus:** {api_info.get('genus', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)


            st.subheader("Informacion Nutricional:")
            nutritions = api_info.get('nutritions', {})
            if nutritions:
                st.markdown('<table class="nutrition-table">', unsafe_allow_html=True)
                st.markdown('<tr><th>Nutrient</th><th>Amount</th></tr>', unsafe_allow_html=True)
                st.markdown(f'<tr><td>Calories</td><td>{nutritions.get("calories", "N/A")}</td></tr>', unsafe_allow_html=True)
                st.markdown(f'<tr><td>Fat</td><td>{nutritions.get("fat", "N/A")}</td></tr>', unsafe_allow_html=True)
                st.markdown(f'<tr><td>Sugar</td><td>{nutritions.get("sugar", "N/A")}</td></tr>', unsafe_allow_html=True)
                st.markdown(f'<tr><td>Carbohydrates</td><td>{nutritions.get("carbohydrates", "N/A")}</td></tr>', unsafe_allow_html=True)
                st.markdown(f'<tr><td>Protein</td><td>{nutritions.get("protein", "N/A")}</td></tr>', unsafe_allow_html=True)
                st.markdown('</table>', unsafe_allow_html=True)
            else:
                st.write("Nutritional information not available.")

        else:
            st.warning("Could not retrieve general fruit information from the API.")


    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure the model file is in the correct directory.")
        st.info("Please make sure you have run the training code and saved the 'modelo_resnet34.pth' file.")
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        st.exception(e)
