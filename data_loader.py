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
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

class DeepfakeDataset(Dataset):
    """Unified dataset class for image-only deepfake detection"""
    
    def __init__(
        self,
        data_paths: List[Tuple[str, int]],  # List of (image_path, label) tuples
        transform=None,
        is_training: bool = True
    ):
        self.data_paths = data_paths
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        img_path, label = self.data_paths[idx]
        
        # Load image
        try:
            if isinstance(img_path, str):
                # Load from file path
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(img_path, Image.Image):
                # Already a PIL Image
                image = np.array(img_path.convert('RGB'))
            else:
                # Assume it's already a numpy array
                image = img_path if isinstance(img_path, np.ndarray) else np.array(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(is_training: bool = True, input_size: int = 224):
    """Get data augmentation transforms for transformers"""
    if is_training and DATA_CONFIG["augmentation"]:
        transform = A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return transform

class HuggingFaceDataLoader:
    """Loader for HuggingFace image datasets"""
    
    @staticmethod
    def load_faceforensics_dataset(dataset_name: str, split: str = "train") -> List[Tuple[str, int]]:
        """Load image dataset from HuggingFace"""
        if load_dataset is None:
            print("Warning: datasets library not available. Install with: pip install datasets")
            return []
        try:
            # Try to load dataset - handle different split options
            try:
                dataset = load_dataset(dataset_name, split=split)
            except:
                # If split doesn't exist, try without split or use 'train'
                try:
                    dataset = load_dataset(dataset_name)
                    if isinstance(dataset, dict):
                        dataset = dataset.get(split, dataset.get('train', list(dataset.values())[0]))
                except:
                    dataset = load_dataset(dataset_name, split='train')
            
            data_paths = []
            
            for item in dataset:
                # Handle different dataset structures
                if 'image' in item:
                    image = item['image']
                    # Get label - try different possible field names
                    label = item.get('label', item.get('labels', item.get('is_fake', 0)))
                    # Convert label to 0/1 if needed
                    if isinstance(label, str):
                        label = 1 if label.lower() in ['fake', 'deepfake', '1', 'true'] else 0
                    data_paths.append((image, int(label)))
                elif 'img' in item:
                    image = item['img']
                    label = item.get('label', item.get('labels', 0))
                    data_paths.append((image, int(label)))
                elif 'path' in item or 'file_name' in item:
                    path = item.get('path', item.get('file_name'))
                    label = item.get('label', item.get('labels', 0))
                    data_paths.append((path, int(label)))
            
            return data_paths
        except Exception as e:
            print(f"Error loading HuggingFace dataset {dataset_name}: {e}")
            print(f"Dataset structure may be different. Check dataset documentation.")
            return []
    
    @staticmethod
    def load_wilddeepfake_dataset(dataset_name: str, split: str = "train") -> List[Tuple[str, int]]:
        """Load WildDeepfake dataset from HuggingFace (images only)"""
        return HuggingFaceDataLoader.load_faceforensics_dataset(dataset_name, split)

def combine_datasets(
    huggingface_paths: List[Tuple[str, int]] = None
) -> List[Tuple[str, int]]:
    """Combine image datasets into one"""
    all_paths = []
    if huggingface_paths:
        all_paths.extend(huggingface_paths)
    return all_paths

def create_dataloaders(
    train_paths: List[Tuple[str, int]],
    val_paths: List[Tuple[str, int]],
    test_paths: List[Tuple[str, int]],
    batch_size: int = 16,
    num_workers: int = 4,
    input_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for image datasets"""
    
    train_transform = get_transforms(is_training=True, input_size=input_size)
    val_transform = get_transforms(is_training=False, input_size=input_size)
    
    train_dataset = DeepfakeDataset(train_paths, transform=train_transform, is_training=True)
    val_dataset = DeepfakeDataset(val_paths, transform=val_transform, is_training=False)
    test_dataset = DeepfakeDataset(test_paths, transform=val_transform, is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def split_dataset(
    data_paths: List[Tuple[str, int]],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split dataset into train, validation, and test sets"""
    import random
    random.shuffle(data_paths)
    
    total = len(data_paths)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_paths = data_paths[:train_end]
    val_paths = data_paths[train_end:val_end]
    test_paths = data_paths[val_end:]
    
    return train_paths, val_paths, test_paths
