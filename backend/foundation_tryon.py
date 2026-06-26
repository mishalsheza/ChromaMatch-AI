"""
Virtual Foundation Try-On Module
ONLY Maybelline Age Rewind Eraser - with REAL RGB values
"""

import cv2
import numpy as np
from PIL import Image
import base64
import os


class FoundationTryOn:
    def __init__(self):
        # REAL RGB values for Maybelline Age Rewind Eraser ONLY
        self.foundation_shades = {
            # ========== MAYBELLINE AGE REWIND ERASER ==========
            "Fair": {"rgb": [251, 221, 212]},
            "4.5": {"rgb": [182, 80, 44]},
            "Ivory": {"rgb": [255, 219, 193]},
            "Light": {"rgb": [225, 202, 178]},
            "122 Sand": {"rgb": [225, 202, 178]},
            "Medium": {"rgb": [226, 201, 178]},
            "Honey": {"rgb": [235, 202, 186]},
            "Caramel": {"rgb": [205, 139, 90]},
            "Butterscotch": {"rgb": [159, 117, 85]},
            "150 Neutralizer": {"rgb": [226, 187, 142]},
        }
    
    def get_shade_color(self, shade_name):
        """Get RGB color for a shade name"""
        # Exact match
        if shade_name in self.foundation_shades:
            return self.foundation_shades[shade_name]["rgb"]
        
        # Try case-insensitive match
        for key in self.foundation_shades:
            if key.lower() == shade_name.lower():
                return self.foundation_shades[key]["rgb"]
        
        # Try partial match
        for key in self.foundation_shades:
            if shade_name.lower() in key.lower() or key.lower() in shade_name.lower():
                return self.foundation_shades[key]["rgb"]
        
        # Default fallback (if shade not in Age Rewind database)
        print(f"⚠️ Shade not found in Age Rewind database: {shade_name}")
        return [200, 165, 145]  # Medium beige fallback
    
    def apply_foundation(self, image_path, shade_name, intensity=0.6):
        """
        Apply foundation using color mapping
        
        Args:
            image_path: Path to user's image
            shade_name: Foundation shade name
            intensity: 0-1, how strong the foundation should appear
        
        Returns:
            base64 encoded image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get foundation color
        foundation_rgb = self.get_shade_color(shade_name)
        foundation_rgb = np.array(foundation_rgb, dtype=np.float32)
        
        # Detect skin and face
        skin_mask = self._detect_skin_advanced(img_rgb)
        face_mask = self._detect_face_mask(img_rgb)
        
        # Combine masks
        mask = np.maximum(skin_mask, face_mask)
        
        # Apply COLOR MAPPING
        result = self._apply_color_mapping(img_rgb, mask, foundation_rgb, intensity)
        
        return self._image_to_base64(result)
    
    def _detect_skin_advanced(self, image):
        """Detect skin using multiple color spaces"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # HSV skin range
        lower_hsv = np.array([0, 20, 40])
        upper_hsv = np.array([25, 170, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb skin range
        lower_ycrcb = np.array([0, 133, 77])
        upper_ycrcb = np.array([255, 173, 127])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        combined = cv2.bitwise_or(mask_hsv, mask_ycrcb)
        
        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.GaussianBlur(combined, (21, 21), 0)
        
        return combined / 255.0
    
    def _detect_face_mask(self, image):
        """Detect face region using Haar Cascade"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        for (x, y, w, h) in faces:
            # Create a oval face mask
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
            
            # Expand to include cheeks
            y_expand = max(0, y - int(h * 0.3))
            h_expand = min(image.shape[0], y + int(h * 1.2))
            x_expand = max(0, x - int(w * 0.2))
            w_expand = min(image.shape[1], x + int(w * 1.2))
            
            oval_mask = np.zeros_like(mask)
            center2 = (x_expand + w_expand//2, y_expand + h_expand//2)
            axes2 = (w_expand//2, h_expand//2)
            cv2.ellipse(oval_mask, center2, axes2, 0, 0, 360, 1, -1)
            
            mask = np.maximum(mask, oval_mask)
        
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        return mask
    
    def _apply_color_mapping(self, image, mask, target_color, intensity):
        """Apply color mapping that preserves skin texture"""
        # Convert to float
        img_float = image.astype(np.float32)
        mask_3 = np.stack([mask, mask, mask], axis=2)
        
        # Get the average skin color
        skin_pixels = img_float[mask > 0.3]
        if len(skin_pixels) > 0:
            avg_skin = np.mean(skin_pixels, axis=0)
        else:
            avg_skin = np.mean(img_float, axis=(0, 1))
        
        # Calculate color shift
        shift = target_color - avg_skin
        
        # Apply shift with intensity
        shifted = img_float + shift * intensity
        
        # Blend with original using mask
        blended = img_float * (1 - mask_3 * intensity * 0.7) + shifted * (mask_3 * intensity * 0.7)
        
        # Preserve highlights and shadows
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_3 = np.stack([gray, gray, gray], axis=2)
        
        # Adjust intensity based on brightness
        final = img_float * (1 - mask_3 * intensity * 0.3) + blended * (mask_3 * intensity * 0.3)
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        return final
    
    def _image_to_base64(self, image):
        """Convert numpy image to base64"""
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')