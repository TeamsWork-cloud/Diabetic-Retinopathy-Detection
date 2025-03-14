import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Check device availability
device = torch.device("cpu")
print(f"Using device: {device}")

# Function to load model from a specified file path
def load_model(model_path, model_class=None, device='cpu'):
    """
    Loads a PyTorch model from a specified file path.

    Parameters:
    model_path - Path to the model file (.pt or .pth)
    model_class - The model class (needed if loading full architecture was not saved)
    device - The device to load the model to ('cuda' or 'cpu')

    Returns:
    model - The loaded model
    model_info - Dictionary containing any additional saved information
    """
    print(f"Loading model from {model_path}...")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Check what's in the checkpoint
    if isinstance(checkpoint, dict):
        checkpoint_keys = checkpoint.keys()
        print(f"Checkpoint contains: {', '.join(checkpoint_keys)}")

        # Extract model and other information
        if 'model' in checkpoint:
            model = checkpoint['model']  # Full model saved
            print("Loaded complete model architecture from checkpoint.")
        elif 'model_state_dict' in checkpoint and model_class is not None:
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Initialized model from class and loaded weights.")
        elif model_class is not None:
            try:
                model = model_class()
                model.load_state_dict(checkpoint)
                print("Loaded state dictionary directly.")
                return model, None
            except:
                print("Couldn't load as direct state dict. Check format or provide correct model class.")
                return None, None
        else:
            print("Error: Model architecture not found in checkpoint and no model_class provided.")
            return None, None

        # Extract other information
        model_info = {key: checkpoint[key] for key in checkpoint_keys if key not in ['model', 'model_state_dict']}
    else:
        if model_class is None:
            print("Error: Direct state dictionary detected but no model_class provided.")
            return None, None

        model = model_class()
        model.load_state_dict(checkpoint)
        model_info = {}
        print("Loaded state dictionary directly.")

    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Print model performance metrics if available
    if 'test_accuracy' in model_info:
        print(f"Model test accuracy: {model_info['test_accuracy']:.2f}%")
    if 'test_f1' in model_info:
        print(f"Model F1 score: {model_info['test_f1']:.2f}%")

    print(f"Model loaded successfully to {device}.")
    return model, model_info

# Function to predict from a single image
def predict(model, image, transform=None, classes=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']):
    """
    Predict diabetic retinopathy class from a single image.

    Parameters:
    model - The trained PyTorch model
    image - PIL image object
    transform - The transformation pipeline
    classes - List of class names

    Returns:
    predicted_class - The predicted class index
    confidence - Confidence score (percentage)
    probabilities - List of class probabilities
    """
    # Set model to evaluation mode
    model.eval()

    try:
        # Convert grayscale to RGB if needed
        if image.mode == 'L':
            print("Converting grayscale image to RGB")
            image = image.convert('RGB')

        # Apply transformations if provided
        if transform:
            img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        else:
            basic_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = basic_transform(image).unsqueeze(0)

        # Move to device
        img_tensor = img_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        conf_value = confidence.item() * 100
        probabilities_list = probabilities.squeeze().tolist()

        return predicted_class, conf_value, probabilities_list

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None, None
