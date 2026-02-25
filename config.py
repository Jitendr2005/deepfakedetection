"""
Configuration file for Deepfake Detection Project (Transformers + Images Only)
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

# Model configuration - Using Transformers
MODEL_CONFIG = {
    "model_name": "vit_base",  # Options: vit_base, vit_large, deit_base, swin_base
    "input_size": 224,
    "num_classes": 2,  # Real vs Fake
    "pretrained": True,
    "dropout": 0.5
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 16,  # Smaller batch size for transformers
    "num_epochs": 50,
    "learning_rate": 2e-5,  # Lower learning rate for transformers
    "weight_decay": 1e-4,
    "num_workers": 4,
    "pin_memory": True,
    "accumulation_steps": 2,  # Gradient accumulation for effective larger batch
    "early_stopping_patience": 10,
    "save_best_only": True,
    "warmup_steps": 500  # Warmup steps for transformers
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

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
