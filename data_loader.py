"""
Data loading utilities for image-only deepfake detection datasets
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import DATA_CONFIG, MODEL_CONFIG

# Optional import
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Optional imports - only import when needed
class DeepfakeDataset(Dataset):
    """Unified dataset class for image-only deepfake detection with lazy loading"""
    
    def __init__(
        self,
        data_paths: List[Tuple[any, int]],  # List of (path/index/Image, label)
        hf_dataset = None,                  # Optional HF dataset object for lazy loading
        transform = None,
        is_training: bool = True
    ):
        self.data_paths = data_paths
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        item, label = self.data_paths[idx]
        
        # Load image
        try:
            if isinstance(item, (str, Path)):
                # Load from file path
                image = cv2.imread(str(item))
                if image is None:
                    raise ValueError(f"Could not load image: {item}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(item, int) and self.hf_dataset is not None:
                # Lazy load from HuggingFace dataset using index
                hf_item = self.hf_dataset[item]
                img_key = 'image' if 'image' in hf_item else 'img'
                pil_img = hf_item[img_key]
                image = np.array(pil_img.convert('RGB'))
            elif isinstance(item, Image.Image):
                # Already a PIL Image
                image = np.array(item.convert('RGB'))
            else:
                # Assume it's already a numpy array
                image = item if isinstance(item, np.ndarray) else np.array(item)
        except Exception as e:
            # Return a black image as fallback to prevent crash during training
            image = np.zeros((MODEL_CONFIG["input_size"], MODEL_CONFIG["input_size"], 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(is_training: bool = True, input_size: int = 224):
    """Get data augmentation transforms"""
    if is_training and DATA_CONFIG["augmentation"]:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class HuggingFaceDataLoader:
    """Loader for HuggingFace image datasets with lazy loading"""
    
    @staticmethod
    def load_faceforensics_dataset(dataset_name: str, split: str = "train", limit: int = None) -> Tuple[List[Tuple[int, int]], any]:
        """Return list of (index, label) and the dataset object itself (Lazy)"""
        if load_dataset is None:
            return [], None
        try:
            dataset = load_dataset(dataset_name, split=split)
            data_indices = []
            
            # Briefly scan labels (don't load images)
            total_items = len(dataset)
            if limit:
                print(f"Sampling {limit} items for metadata...")
                import random
                scan_indices = random.sample(range(total_items), min(limit, total_items))
            else:
                scan_indices = range(total_items)

            for idx in scan_indices:
                item = dataset[idx]
                label = item.get('label', item.get('labels', item.get('is_fake', 0)))
                if isinstance(label, str):
                    label = 1 if label.lower() in ['fake', 'deepfake', '1', 'true'] else 0
                data_indices.append((idx, int(label)))
            
            return data_indices, dataset
        except Exception as e:
            print(f"Error loading HF dataset {dataset_name}: {e}")
            return [], None

def create_dataloaders(
    train_paths: List[Tuple[any, int]],
    val_paths: List[Tuple[any, int]],
    test_paths: List[Tuple[any, int]],
    hf_dataset = None,
    batch_size: int = 16,
    num_workers: int = 2,
    input_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders with lazy loading"""
    
    train_transform = get_transforms(is_training=True, input_size=input_size)
    val_transform = get_transforms(is_training=False, input_size=input_size)
    
    train_dataset = DeepfakeDataset(train_paths, hf_dataset, transform=train_transform)
    val_dataset = DeepfakeDataset(val_paths, hf_dataset, transform=val_transform)
    test_dataset = DeepfakeDataset(test_paths, hf_dataset, transform=val_transform)
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    )

def split_dataset(data_paths, hf_dataset=None, train_split=0.7, val_split=0.15, test_split=0.15):
    """Split dataset with shuffling"""
    import random
    random.shuffle(data_paths)
    total = len(data_paths)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    return data_paths[:train_end], data_paths[train_end:val_end], data_paths[val_end:]
