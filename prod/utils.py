%%writefile utils.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import requests
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np


# Define the class names (must match the order used during object detection training)
# Make sure 'background' is the first class
obj_det_class_names = ['background', 'apple', 'banana', 'orange']
# Define class names for the freshness classifier (must match the order used during classification training)
cls_class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']


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
    except FileNotFoundError:
        print(f"Error: Classification model file not found at {model_path}")
        # Do not raise here, as we might still be able to do object detection
        return None # Return None if model not found

    return model

def load_object_detection_model(model_path, num_classes):
    """Loads the fine-tuned Faster R-CNN model for object detection."""
    # Load a pre-trained model structure
    # Load weights only if model_path exists to avoid redownloading if training hasn't happened
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
    except FileNotFoundError:
        print(f"Error: Object detection model file not found at {model_path}")
        print("Please ensure 'faster_rcnn_fruit_detector.pth' exists and is in the correct directory.")
        # Do not raise here, as we might still be able to run the classification part
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
    # Object detection models often expect normalization after ToTensor
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to tensor and scales to [0, 1]
        # No explicit Resize here, as the model might handle variable sizes
    ])
    # Note: The model expects a list of tensors, even for a single image
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
    num_detection_classes = len(obj_det_class_names) # Number of classes for object detection model
    num_classification_classes = len(cls_class_names) # Number of classes for classification model

    # Load the object detection model
    det_model = load_object_detection_model(detection_model_path, num_detection_classes)
    if det_model is None:
        return [] # Return empty list if detection model not found

    # Load the classification model
    cls_model = load_classification_model(classification_model_path, num_classification_classes)
    if cls_model is None:
        print("Classification model not loaded. Freshness classification will not be performed.")


    # Preprocess the image for object detection
    # The model expects a list of tensors
    image_tensor_list = preprocess_image_object_detection(image)

    # Move tensor and model to the appropriate device (CPU for Streamlit)
    device = torch.device('cpu')
    image_tensor_list = [img.to(device) for img in image_tensor_list]
    det_model.to(device) # Ensure detection model is on the same device
    if cls_model:
        cls_model.to(device) # Ensure classification model is on the same device

    det_model.eval() # Set detection model to evaluation mode
    if cls_model:
        cls_model.eval() # Set classification model to evaluation mode


    detected_objects = []
    with torch.no_grad():
        # Perform object detection
        predictions = det_model(image_tensor_list)

    if predictions:
        prediction = predictions[0] # Get predictions for the first (and only) image

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # Process each detected object
        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                box = boxes[i]
                label_index = labels[i]

                # Ensure label_index is within bounds of obj_det_class_names
                if 0 <= label_index < len(obj_det_class_names):
                     label_name = obj_det_class_names[label_index]

                     # Only include fruit classes, not 'background'
                     if label_name != 'background':
                         obj_dict = {
                             'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])], # Convert to float
                             'label': label_name,
                             'score': float(scores[i]) # Convert to float
                         }

                         # --- Freshness Classification for the detected object ---
                         freshness_prediction = "Freshness prediction not available."
                         if cls_model:
                             try:
                                 # Crop the region of interest from the original image
                                 # Ensure box coordinates are integers for cropping
                                 cropped_image = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3]))).convert('RGB')

                                 # Preprocess the cropped image for classification
                                 processed_cropped_image = preprocess_image_classification(cropped_image)

                                 # Predict freshness
                                 freshness_prediction = predict_classification(cls_model, processed_cropped_image)

                             except Exception as e:
                                 print(f"Error during freshness classification for detected object: {e}")
                                 freshness_prediction = "Error during freshness prediction."

                         obj_dict['freshness_prediction'] = freshness_prediction
                         detected_objects.append(obj_dict)
                else:
                    print(f"Warning: Detected label index {label_index} is out of bounds for obj_det_class_names.")

    return detected_objects


def get_fruit_info_from_api(fruit_name):
    """Fetches general fruit information from the Fruityvice API."""
    try:
        # The API expects lowercase fruit names without "fresh" or "rotten"
        clean_fruit_name = fruit_name.replace('fresh', '').replace('rotten', '')

        # Handle pluralization if necessary, though the API seems to work with singular
        if clean_fruit_name.endswith('s'):
          clean_fruit_name = clean_fruit_name[:-1]

        # Correct the name if necessary (e.g., "apples" -> "apple")
        if clean_fruit_name == 'apples':
             clean_fruit_name = 'apple'
        elif clean_fruit_name == 'bananas':
             clean_fruit_name = 'banana'
        elif clean_fruit_name == 'oranges':
             clean_fruit_name = 'orange'


        api_url = f"https://www.fruityvice.com/api/fruit/{clean_fruit_name.lower()}" # Ensure lowercase
        response = requests.get(api_url)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API for {fruit_name}: {e}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred during API call for {fruit_name}: {e}")
         return None


# No longer needed as freshness prediction is part of detect_fruits result
# def get_fruit_freshness_info(predicted_class_name):
#     """Provides freshness information based on the predicted class."""
#     fruit_info_map = {
#         'freshapples': "This apple appears fresh and ready to eat!",
#         'freshbanana': "This banana is fresh and looks delicious!",
#         'freshoranges': "This orange is fresh and juicy!",
#         'rottenapples': "This apple appears rotten and should not be consumed.",
#         'rottenbanana': "This banana is rotten and not suitable for eating.",
#         'rottenoranges': "This orange is rotten and should be discarded."
#     }
#     return fruit_info_map.get(predicted_class_name, "Could not retrieve freshness information for this fruit.")


if __name__ == '__main__':
    # Example usage (optional, for testing utilities)
    # Assuming you have a test image named 'test_image.jpg' in the same directory
    try:
        test_image_path = 'test_image.jpg' # Replace with a real image path if testing
        dummy_classification_model_path = 'modelo_resnet34.pth' # Replace with your classification model path
        dummy_detection_model_path = 'faster_rcnn_fruit_detector.pth' # Replace with your object detection model path


        # Create dummy model files for testing purposes if they don't exist yet
        if not os.path.exists(dummy_classification_model_path):
            print(f"Creating a dummy classification model file for testing utilities: {dummy_classification_model_path}")
            dummy_cls_model = models.resnet34(pretrained=False)
            dummy_cls_model.fc = nn.Linear(dummy_cls_model.fc.in_features, len(cls_class_names))
            torch.save(dummy_cls_model.state_dict(), dummy_classification_model_path)

        if not os.path.exists(dummy_detection_model_path):
             print(f"Creating a dummy object detection model file for testing utilities: {dummy_detection_model_path}")
             dummy_det_model = fasterrcnn_resnet50_fpn(weights=None) # Load structure without pretrained weights
             in_features = dummy_det_model.roi_heads.box_predictor.cls_score.in_features
             dummy_det_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(obj_det_class_names))
             # Create a dummy state_dict with random weights
             dummy_state_dict = dummy_det_model.state_dict()
             torch.save(dummy_state_dict, dummy_detection_model_path)
             print("Dummy object detection model state dict created.")


        # Load a dummy image or create one for testing
        try:
            test_image = Image.open(test_image_path).convert('RGB')
        except FileNotFoundError:
             print(f"Test image not found at {test_image_path}. Creating a dummy image for testing.")
             # Create a dummy black image
             test_image = Image.new('RGB', (300, 300), color = 'black') # Increased size for potential dummy boxes


        # --- Test Object Detection and Freshness Classification ---
        print("\n--- Testing Object Detection and Freshness Classification ---")
        # Create a dummy image with a "fruit" to test the detection and classification flow
        if test_image.size == (300, 300) and test_image.getpixel((0,0)) == (0,0,0): # If it's the dummy black image
             print("Drawing a dummy box and simulating classification for testing.")
             draw = ImageDraw.Draw(test_image)
             # Draw a dummy box that roughly corresponds to a fruit
             dummy_box = [50, 50, 250, 250]
             draw.rectangle(dummy_box, outline="white", width=5)

             # Simulate a detected object from the dummy box
             simulated_detected_objects = [{
                 'box': dummy_box,
                 'label': 'apple', # Simulate detecting an apple
                 'score': 0.95
             }]

             # To test the freshness classification part, we need to simulate the prediction
             # of the classification model for this cropped region.
             # Since we don't have a real trained detection model here, we can't crop a real detected object.
             # We will simulate the output of the detect_fruits function.

             # Simulate the output as if detect_fruits ran and found an object
             # and then classified its freshness.
             detected_objects_with_freshness = []
             for obj in simulated_detected_objects:
                  # Simulate cropping and classifying the dummy box region
                  try:
                       cropped_img = test_image.crop((int(obj['box'][0]), int(obj['box'][1]), int(obj['box'][2]), int(obj['box'][3]))).convert('RGB')
                       processed_cropped_image = preprocess_image_classification(cropped_img)
                       # Load dummy classification model
                       dummy_cls_model_test = load_classification_model(dummy_classification_model_path, len(cls_class_names))
                       if dummy_cls_model_test:
                            # Simulate a freshness prediction based on the object detection label
                            # In a real scenario, this would come from the cls_model prediction
                            simulated_freshness = 'fresh' + obj['label'] if np.random.rand() > 0.5 else 'rotten' + obj['label'] # Randomly assign fresh/rotten
                            obj['freshness_prediction'] = simulated_freshness
                       else:
                            obj['freshness_prediction'] = "Classification model not loaded for test."

                       detected_objects_with_freshness.append(obj)

                  except Exception as e:
                       print(f"Error simulating cropping/classification for test: {e}")


             detected_objects = detected_objects_with_freshness # Use the simulated results for printing
             print(f"Simulated Detected Objects with Freshness: {detected_objects}")

             # Test API call for a simulated detected object
             if detected_objects:
                  sample_detected_fruit_label = detected_objects[0]['label']
                  api_info = get_fruit_info_from_api(sample_detected_fruit_label)
                  if api_info:
                       print(f"API Info for {sample_detected_fruit_label}: {api_info}")
                  else:
                       print(f"Could not fetch API info for {sample_detected_fruit_label}.")

        else:
            # If a real test image exists, try running the actual detect_fruits function
            print(f"Attempting object detection and classification on test image: {test_image_path}")
            detected_objects = detect_fruits(test_image, detection_model_path=dummy_detection_model_path, classification_model_path=dummy_classification_model_path, confidence_threshold=0.01) # Lower threshold for dummy test
            print(f"Detected Objects with Freshness: {detected_objects}")

            # Test API call for detected objects
            if detected_objects:
                 for i, obj in enumerate(detected_objects):
                      fruit_label = obj['label']
                      api_info = get_fruit_info_from_api(fruit_label)
                      if api_info:
                           print(f"API Info for Detected Fruit {i+1} ({fruit_label}): {api_info}")
                      else:
                           print(f"Could not fetch API info for Detected Fruit {i+1} ({fruit_label}).")


    except FileNotFoundError as e:
        print(f"A required file was not found: {e}")
    except Exception as e:
        print(f"An error occurred during utility testing: {e}")
        import traceback
        traceback.print_exc()
