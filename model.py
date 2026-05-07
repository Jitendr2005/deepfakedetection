"""
Deepfake Detection Models using Transformers and CNNs
Supports Vision Transformer (ViT), DeiT, Swin, EfficientNet, and ResNet architectures
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, DeiTModel, DeiTConfig, AutoImageProcessor
from transformers import SwinModel, SwinConfig
import timm
from config import MODEL_CONFIG

class ViTDeepfakeDetector(nn.Module):
    """Vision Transformer (ViT) based deepfake detector"""
    
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=2, dropout=0.5, pretrained=True):
        super(ViTDeepfakeDetector, self).__init__()
        
        if pretrained:
            self.backbone = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.backbone = ViTModel(config)
        
        # Get hidden size from config
        hidden_size = self.backbone.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # ViT expects input shape: (batch, channels, height, width)
        outputs = self.backbone(pixel_values=x)
        # Use CLS token for classification
        cls_token = outputs.last_hidden_state[:, 0]
        output = self.classifier(cls_token)
        return output

class DeiTDeepfakeDetector(nn.Module):
    """DeiT (Data-efficient Image Transformer) based deepfake detector"""
    
    def __init__(self, model_name="facebook/deit-base-distilled-patch16-224", num_classes=2, dropout=0.5, pretrained=True):
        super(DeiTDeepfakeDetector, self).__init__()
        
        if pretrained:
            self.backbone = DeiTModel.from_pretrained(model_name)
        else:
            config = DeiTConfig.from_pretrained(model_name)
            self.backbone = DeiTModel(config)
        
        hidden_size = self.backbone.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        # DeiT uses distillation token, but we can use CLS token
        cls_token = outputs.last_hidden_state[:, 0]
        output = self.classifier(cls_token)
        return output

class SwinDeepfakeDetector(nn.Module):
    """Swin Transformer based deepfake detector"""
    
    def __init__(self, model_name="microsoft/swin-base-patch4-window7-224", num_classes=2, dropout=0.5, pretrained=True):
        super(SwinDeepfakeDetector, self).__init__()
        
        if pretrained:
            self.backbone = SwinModel.from_pretrained(model_name)
        else:
            config = SwinConfig.from_pretrained(model_name)
            self.backbone = SwinModel(config)
        
        hidden_size = self.backbone.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        # Swin uses pooled output
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        output = self.classifier(pooled_output)
        return output

class CNNDeepfakeDetector(nn.Module):
    """CNN (EfficientNet/ResNet) based deepfake detector using timm"""
    
    def __init__(self, model_name="efficientnet_b0", num_classes=2, dropout=0.5, pretrained=True):
        super(CNNDeepfakeDetector, self).__init__()
        
        # Load backbone from timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get number of input features for the classifier
        if hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Linear):
                in_features = self.backbone.classifier.in_features
            else: # EfficientNet often has classifier as a sequential or similar
                in_features = self.backbone.classifier.in_features if hasattr(self.backbone.classifier, 'in_features') else 1280
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            # Fallback for other timm models
            in_features = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def get_model(model_name: str = "vit_base", num_classes: int = 2, **kwargs):
    """Factory function to get model by name (Transformers or CNNs)"""
    model_name = model_name.lower()
    
    # Transformer Models
    if model_name in ["vit", "vit_base", "vit_base_patch16_224"]:
        return ViTDeepfakeDetector(
            model_name="google/vit-base-patch16-224",
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_name in ["vit_large"]:
        return ViTDeepfakeDetector(
            model_name="google/vit-large-patch16-224",
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_name in ["deit", "deit_base"]:
        return DeiTDeepfakeDetector(
            model_name="facebook/deit-base-distilled-patch16-224",
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_name in ["swin", "swin_base"]:
        return SwinDeepfakeDetector(
            model_name="microsoft/swin-base-patch4-window7-224",
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            pretrained=kwargs.get('pretrained', True)
        )
    
    # CNN Models
    elif model_name in ["efficientnet", "efficientnet_b0"]:
        return CNNDeepfakeDetector(
            model_name="efficientnet_b0",
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_name in ["resnet", "resnet50"]:
        return CNNDeepfakeDetector(
            model_name="resnet50",
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_name in ["xception"]:
        return CNNDeepfakeDetector(
            model_name="xception",
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_name in ["model_a", "cnn_region"]:
        return ModelA(num_classes=num_classes, dropout=kwargs.get('dropout', 0.3))
    elif model_name in ["model_c", "cvit"]:
        return ModelC(num_classes=num_classes, dropout=kwargs.get('dropout', 0.1))
    else:
        raise ValueError(f"Unknown model name: {model_name}. Options: vit_base, vit_base_patch16_224, vit_large, deit_base, swin_base, efficientnet_b0, resnet50, xception, model_a, model_c")

class ModelA(nn.Module):
    """
    12-layer CNN architecture for Eye and Nose regions (Algorithm 1)
    Input size: 50x50
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super(ModelA, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 50x50 -> 25x25 (maxpool 1) -> 12x12 (maxpool 2) -> 6x6 (maxpool 3)
        # 128 * 6 * 6 = 4608
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelC(nn.Module):
    """
    Convolutional Vision Transformer (CViT) for Full Face
    Input size: 224x224
    """
    def __init__(self, num_classes=2, dropout=0.1):
        super(ModelC, self).__init__()
        
        # CNN backbone for initial feature extraction
        # Using a simple CNN as described in CViT papers for deepfake
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Input size 224x224 -> after 3 strides of 2 -> 28x28
        # We treat 28x28 as patches
        
        # Transformer part
        from transformers import ViTConfig, ViTModel
        config = ViTConfig(
            image_size=28,
            patch_size=4, # 28/4 = 7x7 Patches
            num_channels=128,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=512,
            hidden_size=256,
            hidden_dropout_prob=dropout
        )
        self.vit = ViTModel(config)
        
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
    def forward(self, x):
        # Initial CNN features
        x = self.cnn_backbone(x) # Output: (B, 128, 28, 28)
        
        # ViT expects pixel_values, but we can pass embeddings if we wrap it
        # Or more simply, use the transformer directly
        # For CViT, we usually flatten the CNN output to tokens
        
        # Simplified CViT forward:
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits
