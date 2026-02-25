"""
Training script for Deepfake Detection
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

from model import get_model
from data_loader import create_dataloaders, split_dataset
from data_loader import HuggingFaceDataLoader
from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, MODELS_DIR, RESULTS_DIR, DATASET_CONFIG

class Trainer:
    """Training class for deepfake detection"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        
        # Learning rate scheduler with warmup for transformers
        from transformers import get_linear_schedule_with_warmup
        num_training_steps = len(self.train_loader) * TRAIN_CONFIG["num_epochs"]
        warmup_steps = TRAIN_CONFIG.get("warmup_steps", 500)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Also keep ReduceLROnPlateau for validation-based scheduling
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
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
                self.scheduler.step()  # Step transformer scheduler
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
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model: {MODEL_CONFIG['model_name']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate()
            
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
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
                print("✓ Saved best model")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save training history
        self.save_history()
        self.plot_training_history()
        
        return self.history
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, MODELS_DIR / filename)
    
    def save_history(self):
        """Save training history to JSON"""
        history_file = RESULTS_DIR / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()

def load_data():
    """Load and combine image datasets from multiple sources (images only, no videos)"""
    print("Loading image datasets...")
    all_paths = []
    
    # Load HuggingFace image datasets
    try:
        hf_loader = HuggingFaceDataLoader()
        # Use dataset name from config
        dataset_name = DATASET_CONFIG["huggingface"]["faceforensics"]["name"]
        faceforensics_paths = hf_loader.load_faceforensics_dataset(
            dataset_name,
            split="train"
        )
        if faceforensics_paths:
            all_paths.extend(faceforensics_paths)
            print(f"Loaded {len(faceforensics_paths)} samples from {dataset_name}")
    except Exception as e:
        print(f"Warning: Could not load HuggingFace datasets: {e}")
    
    # If no data loaded
    if not all_paths:
        print("No datasets found. Please download datasets first.")
        return None
    
    return all_paths

def main():
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    all_paths = load_data()
    
    if all_paths is None or len(all_paths) == 0:
        print("\n" + "="*60)
        print("IMPORTANT: No datasets found!")
        print("Please download datasets using the download script first.")
        print("See README.md for instructions on downloading datasets.")
        print("="*60)
        return
    
    # Split dataset
    train_paths, val_paths, test_paths = split_dataset(
        all_paths,
        train_split=DATA_CONFIG["train_split"],
        val_split=DATA_CONFIG["val_split"],
        test_split=DATA_CONFIG["test_split"]
    )
    
    print(f"\nDataset splits:")
    print(f"Train: {len(train_paths)} samples")
    print(f"Val: {len(val_paths)} samples")
    print(f"Test: {len(test_paths)} samples")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths,
        val_paths,
        test_paths,
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=TRAIN_CONFIG["num_workers"],
        input_size=MODEL_CONFIG["input_size"]
    )
    
    # Create model
    model = get_model(
        model_name=MODEL_CONFIG["model_name"],
        num_classes=MODEL_CONFIG["num_classes"],
        dropout=MODEL_CONFIG["dropout"],
        pretrained=MODEL_CONFIG["pretrained"]
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Train
    history = trainer.train(num_epochs=TRAIN_CONFIG["num_epochs"])
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
