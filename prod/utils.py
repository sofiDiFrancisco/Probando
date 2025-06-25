
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import os
from pytorch_grad_cam import GradCAM # Import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image # Import utility for visualization

# Define the class names (must match the order used during training)
class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

def load_model(model_path, num_classes):
    """Loads the pre-trained ResNet model."""
    model = models.resnet34(pretrained=False) # Load with pretrained=False as we load state_dict
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load to CPU
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
    return transform(image).unsqueeze(0) # Add batch dimension

def predict_image(model, image_tensor):
    """Makes a prediction using the loaded model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class_index = torch.max(outputs, 1)

    predicted_class_name = class_names[predicted_class_index.item()]
    return predicted_class_name

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
        'freshapples': "This apple appears fresh and ready to eat!",
        'freshbanana': "This banana is fresh and looks delicious!",
        'freshoranges': "This orange is fresh and juicy!",
        'rottenapples': "This apple appears rotten and should not be consumed.",
        'rottenbanana': "This banana is rotten and not suitable for eating.",
        'rottenoranges': "This orange is rotten and should be discarded."
    }
    return fruit_info_map.get(predicted_class_name, "Could not retrieve freshness information for this fruit.")

def generate_gradcam_heatmap(model, target_layer, input_tensor):
    """Generates a Grad-CAM heatmap for a given image and model."""
    # Ensure the model is on the correct device (CPU for Streamlit)
    model.to('cpu')

    # Create a GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False) # use_cuda=False for CPU

    # Generate the heatmap
    # targets can be None to get the highest scoring category
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # In this example, you have only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam


if __name__ == '__main__':
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Example usage (optional, for testing utilities)
    try:
        test_image_path = 'test_image.jpg' # Replace with a real image path if testing
        dummy_model_path = 'modelo_resnet34.pth' # Replace with your model path

        # Create a dummy model for testing purposes if the model file doesn't exist yet
        if not os.path.exists(dummy_model_path):
            print(f"Creating a dummy model file for testing utilities: {dummy_model_path}")
            dummy_model = models.resnet34(pretrained=False)
            dummy_model.fc = nn.Linear(dummy_model.fc.in_features, len(class_names))
            torch.save(dummy_model.state_dict(), dummy_model_path)

        # Load a dummy image or create one for testing
        try:
            test_image = Image.open(test_image_path).convert('RGB')
        except FileNotFoundError:
             print(f"Test image not found at {test_image_path}. Creating a dummy image for testing.")
             # Create a dummy black image
             test_image = Image.new('RGB', (224, 224), color = 'black')

        loaded_model = load_model(dummy_model_path, len(class_names))
        processed_image = preprocess_image(test_image)
        prediction = predict_image(loaded_model, processed_image)
        print(f"Test prediction: {prediction}")

        # Test the new functions
        freshness_info = get_fruit_freshness_info(prediction)
        print(f"Freshness Info: {freshness_info}")

        clean_fruit_name_for_api = prediction.replace('fresh', '').replace('rotten', '')
        if clean_fruit_name_for_api.endswith('s'):
          clean_fruit_name_for_api = clean_fruit_name_for_api[:-1]

        api_info = get_fruit_info_from_api(clean_fruit_name_for_api)
        if api_info:
            print(f"API Info: {api_info}")
        else:
            print("Could not fetch API info.")

        # Test Grad-CAM
        # For ResNet34, a common target layer is the last convolutional layer
        target_layer = loaded_model.layer4[-1]
        print(f"Using target layer: {target_layer}")

        heatmap = generate_gradcam_heatmap(loaded_model, target_layer, processed_image)
        print(f"Generated heatmap with shape: {heatmap.shape}")

        # You can optionally visualize the heatmap on the original image for testing
        # Convert the PIL image to a numpy array (required by show_cam_on_image)
        test_image_np = np.array(test_image)
        visualization = show_cam_on_image(test_image_np.astype(np.float32) / 255., heatmap, use_rgb=True)

        plt.imshow(visualization)
        plt.title("Grad-CAM Visualization (Test)")
        plt.axis('off')
        plt.show()


    except FileNotFoundError:
        print("Dummy model file not found. Cannot run utility test.")
    except Exception as e:
        print(f"An error occurred during utility testing: {e}")
        import traceback
        traceback.print_exc()
