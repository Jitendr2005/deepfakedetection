import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from typing import Dict, Tuple, Optional

class FaceProcessor:
    """
    Handles face detection, alignment, and region extraction (Eyes, Nose)
    as described in the research paper.
    """
    def __init__(self, device='cpu', post_process=True):
        self.device = device
        # Keep all detected faces to choose the best one or use the primary face
        self.mtcnn = MTCNN(
            keep_all=False, 
            device=device, 
            post_process=post_process,
            selection_method='probability'
        )
        
    def detect_and_align(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detects face, returns aligned face, landmarks, and bounding box.
        """
        # Convert BGR to RGB for MTCNN
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        pil_img = Image.fromarray(image_rgb)
        
        # Detect faces and landmarks
        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return None, None, None
            
        # Get the first face (highest probability)
        box = boxes[0]
        landmark = landmarks[0]
        
        # Alignment using landmarks (simple affine transform)
        # Target points for eye alignment
        # We want the eyes to be at roughly [0.3, 0.35] and [0.7, 0.35] of the image
        left_eye = landmark[0]
        right_eye = landmark[1]
        
        # Calculate angle between eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Desired eye position in the aligned face
        desired_left_eye = (0.35, 0.4)
        desired_face_width = 224
        desired_face_height = 224
        
        # Compute the scale
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (0.7 - desired_left_eye[0]) * desired_face_width
        scale = desired_dist / dist
        
        # Compute center between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update center to meet desired position
        tX = desired_face_width * 0.5
        tY = desired_face_height * desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply affine transform
        aligned_face = cv2.warpAffine(image_rgb, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)
        
        # Re-detect landmarks on aligned face or transform them
        # Transforming landmarks is more efficient
        ones = np.ones(shape=(len(landmark), 1))
        points_ones = np.concatenate([landmark, ones], axis=1)
        aligned_landmarks = M.dot(points_ones.T).T
        
        return aligned_face, aligned_landmarks, box

    def extract_regions(self, aligned_face: np.ndarray, aligned_landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts Eye and Nose regions from the aligned face.
        """
        regions = {
            "face": aligned_face
        }
        
        # Aligned landmarks indices (facenet_pytorch/MTCNN standard):
        # 0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth
        
        # 1. Eye region (50x50 crop)
        # Center of both eyes
        eyes_center_x = (aligned_landmarks[0][0] + aligned_landmarks[1][0]) / 2
        eyes_center_y = (aligned_landmarks[0][1] + aligned_landmarks[1][1]) / 2
        
        eye_crop = self._safe_crop(aligned_face, int(eyes_center_x), int(eyes_center_y), 50)
        regions["eyes"] = eye_crop
        
        # 2. Nose region (50x50 crop)
        nose_x = aligned_landmarks[2][0]
        nose_y = aligned_landmarks[2][1]
        
        nose_crop = self._safe_crop(aligned_face, int(nose_x), int(nose_y), 50)
        regions["nose"] = nose_crop
        
        return regions

    def _safe_crop(self, image: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
        """Helper to crop around a center point with padding if necessary"""
        half = size // 2
        x1, y1 = cx - half, cy - half
        x2, y2 = x1 + size, y1 + size
        
        # Handle boundaries
        h, w = image.shape[:2]
        
        # Simple crop with padding if out of bounds
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        
        src_x1, src_y1 = max(0, x1), max(0, y1)
        src_x2, src_y2 = min(w, x2), min(h, y2)
        
        dst_x1, dst_y1 = max(0, -x1), max(0, -y1)
        dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
        
        if src_x1 < src_x2 and src_y1 < src_y2:
            crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
            
        return crop

def process_image_to_regions(image_path: str, processor: FaceProcessor) -> Optional[Dict[str, np.ndarray]]:
    """Convenience function to read image and extract regions"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    aligned_face, landmarks, _ = processor.detect_and_align(image)
    if aligned_face is None:
        return None
        
    return processor.extract_regions(aligned_face, landmarks)
