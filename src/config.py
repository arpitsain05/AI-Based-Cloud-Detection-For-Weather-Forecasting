import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Class Definitions
CLASSES = [
    "Clear Sky",
    "Cloudy",
    "Rain Thunderstorm",
    "Cyclone",
    "Fog",
    "Snow"
]
NUM_CLASSES = len(CLASSES)

# Image Configurations
IMAGE_SIZE = 224  # Standard size for EfficientNet-B0/ResNet-50
# ImageNet normalization statistics (standard for pre-trained torchvision models)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 15
NUM_WORKERS = 4  # Set to 0 on Windows if you experience multi-processing issues

# Training Checkpoints
MODEL_NAME = "efficientnet_b0"  # Options: 'efficientnet_b0', 'resnet50'
BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"best_satellite_weather_{MODEL_NAME}.pth")

# Device Configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

def create_project_structure():
    """Helper function to initialize the necessary directories."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        os.path.join(BASE_DIR, "notebooks"),
        os.path.join(BASE_DIR, "app"),
        os.path.join(PROCESSED_DATA_DIR, "train"),
        os.path.join(PROCESSED_DATA_DIR, "val"),
        os.path.join(PROCESSED_DATA_DIR, "test"),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory verified: {directory}")

if __name__ == "__main__":
    create_project_structure()
