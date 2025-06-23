import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_model(model_path, num_classes, device):
    """Loads the trained PyTorch model state dictionary."""
    model = models.resnet18(weights=None) # Load the model structure
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes) # Adjust the final layer
    model.load_state_dict(torch.load(model_path, map_location=device)) # Load the state dict
    model = model.to(device)
    model.eval() # Set the model to evaluation mode
    return model

def preprocess_image(image: Image.Image, img_size, device):
    """Preprocesses a PIL Image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device) # Apply transforms, add batch dim, move to device
    return image
