import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from model import get_model
from preprocessing import FaceProcessor
import cv2

class DeepfakeEnsemble:
  
    def __init__(self, model_paths: Dict[str, str], device='cpu'):
        self.device = device
        self.models = {}
        
        # Load Model C for Face
        if 'face' in model_paths:
            self.models['face'] = get_model('model_c').to(device)
            self.models['face'].load_state_dict(torch.load(model_paths['face'], map_location=device))
            self.models['face'].eval()
            
        # Load Model A for Eyes
        if 'eyes' in model_paths:
            self.models['eyes'] = get_model('model_a').to(device)
            self.models['eyes'].load_state_dict(torch.load(model_paths['eyes'], map_location=device))
            self.models['eyes'].eval()
            
        # Load Model A/B for Nose
        if 'nose' in model_paths:
            self.models['nose'] = get_model('model_a').to(device)
            self.models['nose'].load_state_dict(torch.load(model_paths['nose'], map_location=device))
            self.models['nose'].eval()
            
        self.processor = FaceProcessor(device=device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict[str, any]:
        """
        Extract regions, run models, and perform majority voting.
        """
        # 1. Preprocess
        aligned_face, landmarks, _ = self.processor.detect_and_align(image)
        if aligned_face is None:
            return {"error": "No face detected"}
            
        regions = self.processor.extract_regions(aligned_face, landmarks)
        
        predictions = {}
        probabilities = {}
        
        # 2. Inference for each region
        for region_name, region_img in regions.items():
            if region_name not in self.models:
                continue
                
            model = self.models[region_name]
            
            # Prepare image (Normalize and ToTensor)
            # Standard normalization for ImageNet or custom? 
            # The research uses standard preprocessing.
            img_tensor = self._preprocess_tensor(region_img, region_name)
            img_tensor = img_tensor.to(self.device).unsqueeze(0)
            
            output = model(img_tensor)
            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            
            predictions[region_name] = pred
            probabilities[region_name] = prob[0].cpu().numpy()
            
        # 3. Majority Voting
        if not predictions:
            return {"error": "No models available for detected regions"}
            
        votes = list(predictions.values())
        final_prediction = max(set(votes), key=votes.count)
        
        # Calculate confidence (average of probabilities for the final class)
        confidences = [probabilities[r][final_prediction] for r in predictions.keys()]
        final_confidence = np.mean(confidences)
        
        return {
            "prediction": final_prediction,
            "confidence": float(final_confidence),
            "region_predictions": predictions,
            "region_probabilities": {k: v.tolist() for k, v in probabilities.items()},
            "extracted_regions": regions # Useful for UI
        }

    def _preprocess_tensor(self, image: np.ndarray, region_name: str) -> torch.Tensor:
        """Simple normalization and conversion to tensor"""
        # Resize if necessary (already done in extraction but being safe)
        target_size = 224 if region_name == 'face' else 50
        if image.shape[0] != target_size:
            image = cv2.resize(image, (target_size, target_size))
            
        # Normalize to 0-1
        image = image.astype(np.float32) / 255.0
        
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)
