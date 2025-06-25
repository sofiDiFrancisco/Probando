import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont # Import ImageDraw and ImageFont
import sys
import os
import time
import requests
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# --- Utility Functions (copied from utils.py to keep app.py self-contained for deployment) ---
# Define the class names (must match the order used during object detection training)
obj_det_class_names = ['background', 'apple', 'banana', 'orange']
# Define class names for the freshness classifier (must match the order used during classification training)
cls_class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']


@st.cache_resource # Cache the model loading
def load_classification_model(model_path, num_classes):
    """Loads the pre-trained ResNet model for classification."""
    model = models.resnet34(pretrained=False) # Load with pretrained=False as we load state_dict
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    try:
        # Load to CPU as Streamlit runs on CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        print(f"Classification model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Classification model file not found at {model_path}")
        return None # Return None if model not found

@st.cache_resource # Cache the model loading
def load_object_detection_model(model_path, num_classes):
    """Loads the fine-tuned Faster R-CNN model for object detection."""
    # Load a pre-trained model structure
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if not os.path.exists(model_path) else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one that has the number of classes we need
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    try:
        # Load the state dictionary
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        print(f"Object detection model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Object detection model file not found at {model_path}")
        print("Please ensure 'faster_rcnn_fruit_detector.pth' exists and is in the correct directory.")
        return None # Return None if model not found

    return model


def preprocess_image_classification(image: Image.Image):
    """Preprocesses the input image for the classification model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension


def preprocess_image_object_detection(image: Image.Image):
    """Preprocesses the input image for the object detection model."""
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to tensor and scales to [0, 1]
    ])
    return [transform(image)]


def predict_classification(model, image_tensor):
    """Makes a prediction using the loaded classification model."""
    if model is None:
        return "Classification model not loaded."
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class_index = torch.max(outputs, 1)

    predicted_class_name = cls_class_names[predicted_class_index.item()]
    return predicted_class_name


def detect_fruits(image: Image.Image, detection_model_path="faster_rcnn_fruit_detector.pth", classification_model_path="modelo_resnet34.pth", confidence_threshold=0.7):
    """
    Detects fruits in an image and classifies their freshness.

    Args:
        image: PIL Image containing fruits.
        detection_model_path: Path to the trained object detection model state_dict.
        classification_model_path: Path to the trained classification model state_dict.
        confidence_threshold: Minimum confidence score for object detection.

    Returns:
        A list of dictionaries, where each dictionary contains 'box', 'label', 'score', and 'freshness_prediction'
        for a detected object. Returns an empty list if the detection model is not found or no objects are detected.
    """
    num_detection_classes = len(obj_det_class_names)
    num_classification_classes = len(cls_class_names)

    # Load the object detection model
    det_model = load_object_detection_model(detection_model_path, num_detection_classes)
    if det_model is None:
        return []

    # Load the classification model
    cls_model = load_classification_model(classification_model_path, num_classification_classes)
    if cls_model is None:
        print("Classification model not loaded. Freshness classification will not be performed.")


    # Preprocess the image for object detection
    image_tensor_list = preprocess_image_object_detection(image)

    # Move tensor and model to the appropriate device (CPU for Streamlit)
    device = torch.device('cpu')
    image_tensor_list = [img.to(device) for img in image_tensor_list]
    det_model.to(device)
    if cls_model:
        cls_model.to(device)

    det_model.eval()
    if cls_model:
        cls_model.eval()


    detected_objects = []
    with torch.no_grad():
        predictions = det_model(image_tensor_list)

    if predictions:
        prediction = predictions[0]

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                box = boxes[i]
                label_index = labels[i]

                if 0 <= label_index < len(obj_det_class_names):
                     label_name = obj_det_class_names[label_index]

                     if label_name != 'background':
                         obj_dict = {
                             'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                             'label': label_name,
                             'score': float(scores[i])
                         }

                         freshness_prediction = "Freshness prediction not available."
                         if cls_model:
                             try:
                                 # Ensure box coordinates are within image bounds for cropping
                                 img_width, img_height = image.size
                                 cropped_box = (
                                     max(0, int(box[0])),
                                     max(0, int(box[1])),
                                     min(img_width, int(box[2])),
                                     min(img_height, int(box[3]))
                                 )
                                 cropped_image = image.crop(cropped_box).convert('RGB')

                                 # Ensure cropped image is not empty
                                 if cropped_image.size[0] > 0 and cropped_image.size[1] > 0:
                                     # Preprocess the cropped image for classification
                                     processed_cropped_image = preprocess_image_classification(cropped_image)
                                     # Predict freshness
                                     freshness_prediction = predict_classification(cls_model, processed_cropped_image)
                                 else:
                                     freshness_prediction = "Could not crop valid region for freshness."


                             except Exception as e:
                                 print(f"Error during freshness classification for detected object: {e}")
                                 freshness_prediction = "Error during freshness prediction."

                         obj_dict['freshness_prediction'] = freshness_prediction
                         detected_objects.append(obj_dict)
                else:
                    print(f"Warning: Detected label index {label_index} is out of bounds for obj_det_class_names.")

    return detected_objects


@st.cache_data(ttl=3600) # Cache API responses for 1 hour
def get_fruit_info_from_api(fruit_name):
    """Fetches general fruit information from the Fruityvice API."""
    try:
        clean_fruit_name = fruit_name.replace('fresh', '').replace('rotten', '')

        if clean_fruit_name.endswith('s'):
          clean_fruit_name = clean_fruit_name[:-1]

        if clean_fruit_name == 'apples':
             clean_fruit_name = 'apple'
        elif clean_fruit_name == 'bananas':
             clean_fruit_name = 'banana'
        elif clean_fruit_name == 'oranges':
             clean_fruit_name = 'orange'

        api_url = f"https://www.fruityvice.com/api/fruit/{clean_fruit_name.lower()}" # Ensure lowercase
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API for {fruit_name}: {e}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred during API call for {fruit_name}: {e}")
         return None


# --- Streamlit App Code ---
st.set_page_config(
    page_title="FruitAI - Object Detector & Freshness Detector",
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
st.markdown('<div class="header">🍏 FruitAI Object Detector & Freshness Detector 🍌🍊</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an image to detect multiple fruits and check their freshness</div>', unsafe_allow_html=True)

# Barra lateral con información
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/415/415733.png", width=100)
    st.markdown("## About")
    st.info("""
        This application attempts to detect multiple apples, bananas, and oranges in an image
        using a fine-tuned object detection model (Faster R-CNN).
        For each detected fruit, it classifies its freshness (fresh or rotten) using a separate
        classification model and fetches general fruit information from the Fruityvice API.
        Developed by [Your Name/Team Name].
    """)
    st.markdown("## Datasets & Models")
    st.info("""
    - Object Detection Model: Fine-tuned Faster R-CNN on a custom/simulated dataset.
    - Freshness Classification Model: Fine-tuned ResNet34 on the Fruits Fresh and Rotten dataset.
    - General Fruit Info: [Fruityvice API](https://www.fruityvice.com/)
    """)
    st.markdown("---")
    st.write("© 2024 FruitAI")


# Main content area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image.", use_column_width=True)

        with col2:
            st.write("")
            st.write("Processing image...")

            with st.spinner('Detecting objects and classifying freshness...'):
                detection_model_path = "faster_rcnn_fruit_detector.pth"
                classification_model_path = "modelo_resnet34.pth"
                # Adjust confidence threshold if needed
                detected_objects = detect_fruits(image, detection_model_path=detection_model_path, classification_model_path=classification_model_path, confidence_threshold=0.6)
                time.sleep(1)

            st.subheader("Detection and Classification Results:")

            if detected_objects:
                st.write(f"Found {len(detected_objects)} potential fruits:")

                # Draw bounding boxes on the image with labels, score, and freshness
                draw_image = image.copy()
                draw = ImageDraw.Draw(draw_image)
                try:
                    # Try loading a common font, fall back to default
                    font_path = "arial.ttf" # Common font on many systems
                    font = ImageFont.truetype(font_path, 20)
                except IOError:
                    font = ImageFontFont.load_default()


                for obj in detected_objects:
                    box = obj['box']
                    label = obj['label']
                    score = obj['score']
                    freshness = obj.get('freshness_prediction', 'N/A') # Get freshness, default to N/A

                    # Determine box color based on freshness
                    box_color = "red" # Default color
                    if 'fresh' in freshness.lower():
                        box_color = "green"
                    elif 'rotten' in freshness.lower():
                        box_color = "orange" # Use orange for rotten boxes

                    draw.rectangle(box, outline=box_color, width=3)

                    # Prepare text for bounding box
                    display_text = f"{label.capitalize()}: {score:.2f}"
                    if freshness != 'N/A' and freshness != "Freshness prediction not available." and not freshness.startswith("Error"):
                         # Clean up freshness string for display
                         cleaned_freshness = freshness.replace(label, '').replace('fresh', 'FRESH').replace('rotten', 'ROTTEN').strip().upper()
                         display_text += f" ({cleaned_freshness})"

                    # Position text slightly above the box
                    text_x = box[0]
                    # Estimate text height and position text above the box
                    try:
                        text_width, text_height = draw.textsize(display_text, font=font)
                    except AttributeError: # Fallback for older PIL versions
                         text_width, text_height = font.getsize(display_text)

                    text_y = box[1] - text_height - 5 if box[1] - text_height - 5 > 0 else box[1] + 5


                    # Draw text with a background for readability
                    # Add a small padding to the text background box
                    text_bg_box = (text_x, text_y, text_x + text_width + 5, text_y + text_height + 5) # Add padding
                    draw.rectangle(text_bg_box, fill=box_color) # Draw background rectangle
                    draw.text((text_x + 2, text_y + 2), display_text, fill="white", font=font) # Draw text with white color


                st.image(draw_image, caption="Detected Fruits with Freshness.", use_column_width=True)

                st.markdown("---") # Separator

                st.subheader("Information for Each Detected Fruit:")
                for i, obj in enumerate(detected_objects):
                    label = obj['label']
                    score = obj['score']
                    freshness = obj.get('freshness_prediction', 'N/A')
                    box = obj['box']

                    # Use an expander for each fruit's details
                    with st.expander(f"Fruit {i+1}: {label.capitalize()} (Confidence: {score:.2f})"):

                        # Display freshness with visual cues
                        st.write("**Freshness:** ", end="")
                        if 'fresh' in freshness.lower():
                            st.markdown(f'<span style="color: green;">{freshness} ✨</span>', unsafe_allow_html=True)
                        elif 'rotten' in freshness.lower():
                             st.markdown(f'<span style="color: red;">{freshness} 🤢</span>', unsafe_allow_html=True)
                        else:
                             st.write(freshness) # Display as plain text if unknown

                        # Get General Fruit Information from API
                        api_info = get_fruit_info_from_api(label)

                        if api_info:
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown(f"**General Info for {api_info.get('name', 'N/A').capitalize()}:**", unsafe_allow_html=True)
                            st.write(f"**Family:** {api_info.get('family', 'N/A')}")
                            st.write(f"**Order:** {api_info.get('order', 'N/A')}")
                            st.write(f"**Genus:** {api_info.get('genus', 'N/A')}")
                            st.markdown('</div>', unsafe_allow_html=True)


                            st.subheader("Nutritional Information:")
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
                            st.warning(f"Could not retrieve general fruit information from the API for {label}.")

                    st.markdown("---")


            else:
                st.info("No fruits detected in the image with the current confidence threshold.")
                st.warning("Please ensure the 'faster_rcnn_fruit_detector.pth' model file exists and is in the correct directory and try adjusting the confidence threshold.")


    except FileNotFoundError as e:
        st.error(f"A required model file was not found: {e}")
        st.info("Please ensure you have trained and saved the necessary model files ('modelo_resnet34.pth' and 'faster_rcnn_fruit_detector.pth').")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.exception(e)
