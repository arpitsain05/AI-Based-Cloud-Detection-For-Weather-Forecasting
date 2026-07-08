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
    "cloudy",
    "cyclone",
    "rainy",
    "shine",
    "sunrise"
]
NUM_CLASSES = len(CLASSES)

# Image Configurations
IMAGE_SIZE = 224  # Standard size for EfficientNet-B0/ResNet-50
# Calculated custom dataset normalization statistics (mean and std of training set)
NORM_MEAN = [0.4611, 0.4577, 0.4499]
NORM_STD = [0.2748, 0.2562, 0.2911]

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 15
NUM_WORKERS = 0  # Set to 0 on Windows to prevent multi-processing issues

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
