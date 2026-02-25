"""
Deepfake Detection Models using Transformers
Supports Vision Transformer (ViT), DeiT, and other transformer architectures
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, DeiTModel, DeiTConfig, AutoImageProcessor
from transformers import SwinModel, SwinConfig
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

def get_model(model_name: str = "vit_base", num_classes: int = 2, **kwargs):
    """Factory function to get transformer model by name"""
    model_name = model_name.lower()
    
    if model_name in ["vit", "vit_base"]:
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
    else:
        raise ValueError(f"Unknown model name: {model_name}. Options: vit_base, vit_large, deit_base, swin_base")
