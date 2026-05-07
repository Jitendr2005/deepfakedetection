"""
Training script for Deepfake Detection
Supports multiple model architectures
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

from model import get_model
from data_loader import create_dataloaders, split_dataset
from data_loader import HuggingFaceDataLoader
from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, MODELS_DIR, RESULTS_DIR, DATASET_CONFIG

class Trainer:
    """Training class for deepfake detection"""
    
    def __init__(self, model, model_name, train_loader, val_loader, device):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Determine if it's a transformer or CNN for scheduling/LR
        self.is_transformer = "vit" in model_name or "deit" in model_name or "swin" in model_name
        
        # Set learning rate based on model type
        lr = TRAIN_CONFIG["learning_rate"] if self.is_transformer else TRAIN_CONFIG["cnn_learning_rate"]
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        
        # Schedulers
        if self.is_transformer:
            from transformers import get_linear_schedule_with_warmup
            num_training_steps = len(self.train_loader) * TRAIN_CONFIG["num_epochs"]
            warmup_steps = TRAIN_CONFIG.get("warmup_steps", 500)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = None

        # Plateau scheduler for all models
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        accumulation_steps = TRAIN_CONFIG.get("accumulation_steps", 1)
        
        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc="Training")
        
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item() * accumulation_steps
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model Architecture: {self.model_name}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Update learning rate (plateau scheduler for validation-based adjustment)
            self.plateau_scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_model(f"best_{self.model_name}.pth")
                print(f"✓ Saved best {self.model_name} model")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
                print(f"\nEarly stopping triggered!")
                break
        
        # Save training history
        self.save_history()
        self.plot_training_history()
        
        return self.history
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, MODELS_DIR / filename)
    
    def save_history(self):
        """Save training history to JSON"""
        history_file = RESULTS_DIR / f"history_{self.model_name}.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_title(f'Loss - {self.model_name}')
        axes[0].legend()
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_title(f'Accuracy - {self.model_name}')
        axes[1].legend()
        
        plt.savefig(RESULTS_DIR / f"history_{self.model_name}.png")
        plt.close()

def load_data(limit=None):
    """Load image datasets with lazy loading"""
    print("Loading image datasets...")
    all_paths = []
    hf_dataset = None
    try:
        hf_loader = HuggingFaceDataLoader()
        dataset_name = DATASET_CONFIG["huggingface"]["faceforensics"]["name"]
        res = hf_loader.load_faceforensics_dataset(dataset_name, split="train", limit=limit)
        if res:
            all_paths, hf_dataset = res
            print(f"Loaded metadata for {len(all_paths)} samples (Lazy Loading)")
    except Exception as e:
        print(f"Warning: {e}")
    return all_paths, hf_dataset

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument("--model", type=str, default=MODEL_CONFIG["model_name"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_epochs"], help="Number of epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for toy run")
    parser.add_argument("--dry-run", action="store_true", help="Just check environment and exit")
    args = parser.parse_args()

    from config import DEVICE
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    if args.dry_run:
        print("Dry run successful! Environment is ready.")
        import timm
        import datasets
        print(f"Timm version: {timm.__version__}")
        print(f"Datasets version: {datasets.__version__}")
        return
    
    # Load data
    all_paths, hf_dataset = load_data(limit=args.limit)
    
    if not all_paths:
        print("No datasets found! Run download_datasets.py first.")
        return
    
    # Apply limit for toy run
    if args.limit:
        print(f"Applying sample limit: {args.limit}")
        import random
        random.seed(42)  # For reproducibility during demo
        random.shuffle(all_paths)
        all_paths = all_paths[:args.limit]
    
    # Split dataset
    train_paths, val_paths, test_paths = split_dataset(
        all_paths,
        hf_dataset=hf_dataset,
        train_split=DATA_CONFIG["train_split"],
        val_split=DATA_CONFIG["val_split"],
        test_split=DATA_CONFIG["test_split"]
    )
    
    print(f"\nDataset splits: Train={len(train_paths)}, Val={len(val_paths)}")
    
    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(
        train_paths,
        val_paths,
        test_paths,
        hf_dataset=hf_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=TRAIN_CONFIG["num_workers"],
        input_size=MODEL_CONFIG["input_size"]
    )
    
    # Create model
    model = get_model(
        model_name=args.model,
        num_classes=MODEL_CONFIG["num_classes"],
        dropout=MODEL_CONFIG["dropout"],
        pretrained=MODEL_CONFIG["pretrained"]
    )
    
    # Create trainer
    trainer = Trainer(model, args.model, train_loader, val_loader, device)
    
    # Train
    trainer.train(num_epochs=args.epochs)

if __name__ == "__main__":
    main()
