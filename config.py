"""
Configuration file for Deepfake Detection Project
Supports both Transformer and CNN architectures
"""
import os
import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_name": "efficientnet_b0",  # Default model
    "available_models": ["vit_base", "efficientnet_b0", "resnet50", "swin_base"],
    "input_size": 224,
    "num_classes": 2,  # Real vs Fake
    "pretrained": True,
    "dropout": 0.5
}

# Training configuration - Model specific defaults
TRAIN_CONFIG = {
    "batch_size": 16,
    "num_epochs": 50,
    "learning_rate": 2e-5,  # Default for transformers
    "cnn_learning_rate": 1e-4,  # Default for CNNs
    "weight_decay": 1e-4,
    "num_workers": 2,  # Reduced for local machine stability
    "pin_memory": True,
    "accumulation_steps": 2,
    "early_stopping_patience": 10,
    "save_best_only": True,
    "warmup_steps": 500
}

# Data configuration - Images Only
DATA_CONFIG = {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "augmentation": True,
    "normalize": True,
    "image_only": True  # Only images, no videos
}

# Dataset paths - HuggingFace image datasets only
# Note: Use actual available datasets from HuggingFace
DATASET_CONFIG = {
    "huggingface": {
        "faceforensics": {
            "name": "JamieWithofs/Deepfake-and-real-images-4",  # 210k images
            "path": DATA_DIR / "huggingface" / "faceforensics"
        },
        "wilddeepfake": {
            "name": "Hemg/deepfake-and-real-images",  # 190k images
            "path": DATA_DIR / "huggingface" / "wilddeepfake"
        }
    }
}

# Device configuration - Improved for Mac support
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
