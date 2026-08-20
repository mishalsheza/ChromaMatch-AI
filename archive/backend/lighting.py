import cv2
import numpy as np

def apply_gray_world_white_balance(image):
    """
    Applies Gray World white balance to a BGR image.
    Assumes that the average color in a scene is gray.
    """
    # Convert to float to avoid overflow
    img_float = image.astype(np.float32)
    
    # Calculate average of B, G, R channels
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])
    
    # Avoid division by zero
    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        return image
        
    # Calculate scale factors
    gray = (avg_b + avg_g + avg_r) / 3.0
    scale_b = gray / avg_b
    scale_g = gray / avg_g
    scale_r = gray / avg_r
    
    # Scale channels
    img_float[:, :, 0] *= scale_b
    img_float[:, :, 1] *= scale_g
    img_float[:, :, 2] *= scale_r
    
    # Clip values to [0, 255] and convert back to uint8
    balanced = np.clip(img_float, 0, 255).astype(np.uint8)
    return balanced

def apply_clahe(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel of LAB.
    Balances exposure and light variations without shifting colors.
    """
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back and convert to BGR
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def correct_lighting(image):
    """
    Full pipeline to balance lighting and colors.
    """
    # 1. Apply White Balance
    wb_img = apply_gray_world_white_balance(image)
    # 2. Apply CLAHE
    corrected = apply_clahe(wb_img)
    return corrected
