"""
Evaluation script for Deepfake Detection
"""
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

from model import get_model
from config import MODEL_CONFIG, MODELS_DIR, RESULTS_DIR

def load_model_checkpoint(model_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    model = get_model(
        model_name=MODEL_CONFIG["model_name"],
        num_classes=MODEL_CONFIG["num_classes"],
        dropout=MODEL_CONFIG["dropout"],
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def main():
    """Main evaluation function"""
    import sys
    
    # Get model path
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        model_files = list(MODELS_DIR.glob("*.pth"))
        if not model_files:
            print("No model found! Please train a model first.")
            return
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"Using model: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model_checkpoint(model_path, device)
    print("Model loaded successfully!")
    print("\nNote: To evaluate on test data, load test dataset and run predictions.")
    print("See train.py for example of how to load datasets.")

if __name__ == "__main__":
    main()
