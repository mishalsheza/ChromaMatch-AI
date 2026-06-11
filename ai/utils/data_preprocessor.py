"""
Data preprocessing for AI training
"""
import cv2
import numpy as np
import os
import mediapipe as mp

class DataPreprocessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
    
    def extract_cheek_ai(self, image):
        """Extract cheek regions using AI landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape
            
            # Cheek landmark indices from MediaPipe
            left_indices = [117, 118, 119, 120, 121, 122, 123, 124]
            right_indices = [346, 347, 348, 349, 350, 351, 352, 353]
            
            # Get cheek regions
            left_cheek = self._get_cheek_region(image, landmarks, left_indices, w, h)
            right_cheek = self._get_cheek_region(image, landmarks, right_indices, w, h)
            
            return left_cheek, right_cheek, True
        
        return None, None, False
    
    def _get_cheek_region(self, image, landmarks, indices, w, h):
        """Extract and normalize cheek region"""
        points = []
        for idx in indices:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            points.append([x, y])
        
        points = np.array(points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        cheek = image[y_min:y_max, x_min:x_max]
        if cheek.size > 0:
            cheek = cv2.resize(cheek, (128, 128))
            return cheek
        return None
    
    def augment_image(self, image):
        """Create augmented versions for training"""
        augmented = [image]
        
        # Horizontal flip
        augmented.append(cv2.flip(image, 1))
        
        # Brightness variations
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for factor in [0.8, 1.2]:
            hsv_copy = hsv.copy()
            hsv_copy[:, :, 2] = np.clip(hsv_copy[:, :, 2] * factor, 0, 255)
            augmented.append(cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR))
        
        return augmented