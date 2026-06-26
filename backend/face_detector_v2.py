"""
face_detector_v2.py — Multi-region skin sampling with quality validation
Uses Haar Cascade for face detection (no MediaPipe dependency)
"""

import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        # Use Haar Cascade for face detection (no MediaPipe dependency)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Region indices for facial landmarks (using ratios instead of MediaPipe)
        self.region_ratios = {
            "left_cheek": (0.15, 0.50, 0.30, 0.30),   # x, y, w, h ratios
            "right_cheek": (0.55, 0.50, 0.30, 0.30),
            "jawline_left": (0.05, 0.65, 0.20, 0.20),
            "jawline_right": (0.75, 0.65, 0.20, 0.20),
            "chin": (0.35, 0.80, 0.30, 0.15),
            "forehead": (0.15, 0.10, 0.70, 0.30),
        }
        
        # Weights for weighted average (cheeks + jawline focused)
        self.weights = {
            "left_cheek": 0.30,
            "right_cheek": 0.30,
            "jawline_left": 0.15,
            "jawline_right": 0.15,
            "chin": 0.10,
            "forehead": 0.00,  # Excluded from foundation matching
        }
        
        # Quality thresholds
        self.MIN_BRIGHTNESS = 30
        self.MAX_BRIGHTNESS = 230
        self.MIN_LIGHTNESS = 35
        self.MAX_LIGHTNESS = 225
        self.MAX_LIGHTING_DIFF = 35
        self.MIN_FACE_RATIO = 0.08
        self.MIN_QUALITY_SCORE = 30
        self.MAX_QUALITY_DIFF_WEIGHT = 20
    
    def detect_face_regions(self, image):
        """Detect face and extract regions using Haar Cascade"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return None, False, "No face detected", None
        
        # Get the largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_box = (x, y, fw, fh)
        
        region_colors = {}
        
        for region_name, (rx, ry, rw, rh) in self.region_ratios.items():
            # Calculate region coordinates
            rx1 = int(x + rx * fw)
            ry1 = int(y + ry * fh)
            rx2 = int(rx1 + rw * fw)
            ry2 = int(ry1 + rh * fh)
            
            # Clamp to image boundaries
            rx1 = max(0, rx1)
            ry1 = max(0, ry1)
            rx2 = min(w, rx2)
            ry2 = min(h, ry2)
            
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            
            region = image[ry1:ry2, rx1:rx2]
            if region.size > 0:
                lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
                L = lab[:, :, 0]
                mask = (L > self.MIN_LIGHTNESS) & (L < self.MAX_LIGHTNESS)
                
                valid_pixels = lab[mask]
                if len(valid_pixels) > 10:
                    avg_lab = np.median(valid_pixels, axis=0)
                    region_colors[region_name] = avg_lab
                else:
                    avg_lab = np.mean(lab, axis=(0, 1))
                    region_colors[region_name] = avg_lab
        
        return region_colors, True, None, face_box

    def scan_full_face(self, image):
        """
        Scan the full visible face for stable skin color.

        Returns:
            result, success, error

        result includes:
            - median_skin_color: RGB median after outlier rejection
            - mean_skin_color: RGB mean after outlier rejection
            - std_dev: RGB/LAB standard deviation and a 0-100 uniformity score
            - quality_score: 0-100 score based on valid pixel coverage and uniformity
            - valid_pixel_count / total_face_pixels / face_box for debugging
        """
        if image is None or image.size == 0:
            return None, False, "Invalid image"

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return None, False, "No face detected"

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_box = (int(x), int(y), int(fw), int(fh))

        face_mask = self._build_full_face_skin_mask(image, face_box)
        total_face_pixels = int(np.count_nonzero(face_mask))

        if total_face_pixels == 0:
            return None, False, "Could not isolate face skin area"

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        L = lab[:, :, 0]

        # Hard L* rejection removes deep shadows and blown highlights.
        valid_mask = face_mask & (L >= 30) & (L <= 230)

        lab_pixels = lab[valid_mask].astype(np.float32)
        rgb_pixels = rgb[valid_mask].astype(np.float32)

        if len(lab_pixels) < 100:
            return None, False, "Not enough valid skin pixels after masking"

        # Percentile trimming removes color casts, shadows, and saturated spots.
        keep = np.ones(len(lab_pixels), dtype=bool)
        for channel in range(3):
            low, high = np.percentile(lab_pixels[:, channel], [10, 90])
            keep &= (lab_pixels[:, channel] >= low) & (lab_pixels[:, channel] <= high)

        lab_pixels = lab_pixels[keep]
        rgb_pixels = rgb_pixels[keep]

        if len(lab_pixels) < 100:
            return None, False, "Not enough valid skin pixels after percentile filtering"

        # Standard deviation filtering in LAB removes remaining outliers.
        mean_lab = np.mean(lab_pixels, axis=0)
        std_lab = np.std(lab_pixels, axis=0)
        safe_std_lab = np.where(std_lab < 1.0, 1.0, std_lab)
        z_scores = np.abs((lab_pixels - mean_lab) / safe_std_lab)
        keep = np.all(z_scores <= 2.0, axis=1)

        filtered_lab = lab_pixels[keep]
        filtered_rgb = rgb_pixels[keep]

        if len(filtered_lab) < 100:
            return None, False, "Not enough valid skin pixels after outlier removal"

        median_rgb = np.median(filtered_rgb, axis=0)
        mean_rgb = np.mean(filtered_rgb, axis=0)
        std_rgb = np.std(filtered_rgb, axis=0)
        median_lab = np.median(filtered_lab, axis=0)
        mean_lab = np.mean(filtered_lab, axis=0)
        std_lab = np.std(filtered_lab, axis=0)

        valid_pixel_count = int(len(filtered_rgb))
        coverage_ratio = valid_pixel_count / max(total_face_pixels, 1)

        # Lower color spread means more even skin sampling.
        chroma_spread = float(np.mean(std_lab[1:]))
        lightness_spread = float(std_lab[0])
        uniformity_score = 100 - ((lightness_spread * 1.2) + (chroma_spread * 2.0))
        uniformity_score = int(max(0, min(100, round(uniformity_score))))

        coverage_score = min(100, coverage_ratio * 180)
        pixel_score = min(100, valid_pixel_count / 150.0)
        quality_score = int(round(
            (coverage_score * 0.45) + (pixel_score * 0.25) + (uniformity_score * 0.30)
        ))
        quality_score = max(0, min(100, quality_score))

        result = {
            "median_skin_color": {
                "rgb": np.clip(median_rgb, 0, 255).round(2).tolist(),
                "lab": np.clip(median_lab, 0, 255).round(2).tolist(),
            },
            "mean_skin_color": {
                "rgb": np.clip(mean_rgb, 0, 255).round(2).tolist(),
                "lab": np.clip(mean_lab, 0, 255).round(2).tolist(),
            },
            "std_dev": {
                "rgb": std_rgb.round(2).tolist(),
                "lab": std_lab.round(2).tolist(),
                "skin_uniformity_score": uniformity_score,
            },
            "quality_score": quality_score,
            "valid_pixel_count": valid_pixel_count,
            "total_face_pixels": total_face_pixels,
            "coverage_ratio": round(float(coverage_ratio), 4),
            "face_box": {
                "x": int(x),
                "y": int(y),
                "w": int(fw),
                "h": int(fh),
            },
            "method": "full_face_lab_outlier_scan",
        }

        return result, True, None

    def _build_full_face_skin_mask(self, image, face_box):
        """Build a full-face mask while excluding common non-skin features."""
        h, w = image.shape[:2]
        x, y, fw, fh = face_box

        mask = np.zeros((h, w), dtype=np.uint8)

        center = (int(x + fw * 0.50), int(y + fh * 0.53))
        axes = (int(fw * 0.43), int(fh * 0.50))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Remove hair-prone upper face band.
        self._clear_rect(mask, x + fw * 0.05, y, fw * 0.90, fh * 0.18)

        # Remove eyebrows and eyes.
        self._clear_rect(mask, x + fw * 0.16, y + fh * 0.23, fw * 0.27, fh * 0.16)
        self._clear_rect(mask, x + fw * 0.57, y + fh * 0.23, fw * 0.27, fh * 0.16)
        self._clear_rect(mask, x + fw * 0.12, y + fh * 0.31, fw * 0.32, fh * 0.15)
        self._clear_rect(mask, x + fw * 0.56, y + fh * 0.31, fw * 0.32, fh * 0.15)

        # Remove lips and moustache/shadow-prone area around the mouth.
        self._clear_rect(mask, x + fw * 0.28, y + fh * 0.63, fw * 0.44, fh * 0.18)

        # Remove outer edge where background, ears, sideburns, and hair leak in.
        self._clear_rect(mask, x, y, fw * 0.10, fh)
        self._clear_rect(mask, x + fw * 0.90, y, fw * 0.10, fh)

        skin_color_mask = self._skin_color_mask(image)
        mask = cv2.bitwise_and(mask, skin_color_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask.astype(bool)

    def _skin_color_mask(self, image):
        """Broad skin-color mask used only after the face geometry mask."""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Y, Cr, Cb = cv2.split(ycrcb)
        H, S, V = cv2.split(hsv)

        ycrcb_mask = (
            (Y > 35) & (Y < 245) &
            (Cr > 130) & (Cr < 180) &
            (Cb > 75) & (Cb < 145)
        )

        hsv_mask = (
            (H < 35) &
            (S > 10) & (S < 180) &
            (V > 35) & (V < 245)
        )

        return ((ycrcb_mask | hsv_mask).astype(np.uint8)) * 255

    def _clear_rect(self, mask, x, y, width, height):
        """Clear a rectangle from a uint8 mask with image-boundary clamping."""
        h, w = mask.shape[:2]
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(w, int(round(x + width)))
        y2 = min(h, int(round(y + height)))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
    
    def _extract_region(self, image, points, padding=12):
        """Extract a patch around points"""
        points = np.array(points)
        x_min = max(0, points[:, 0].min() - padding)
        x_max = min(image.shape[1], points[:, 0].max() + padding)
        y_min = max(0, points[:, 1].min() - padding)
        y_max = min(image.shape[0], points[:, 1].max() + padding)
        
        if x_max <= x_min or y_max <= y_min:
            return None
        
        return image[y_min:y_max, x_min:x_max]
    
    def get_average_skin_color(self, image):
        """Get weighted average skin color from all regions"""
        h, w = image.shape[:2]
        total_pixels = h * w
        
        # Image quality validation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        print(f"🔍 DEBUG: Brightness = {brightness:.1f}")
        
        if brightness < self.MIN_BRIGHTNESS:
            print(f"❌ Rejected: Image too dark (brightness: {brightness:.1f} < {self.MIN_BRIGHTNESS})")
            return None, {
                "error": "Image too dark. Please use better lighting.",
                "brightness": brightness,
                "quality_score": 0,
                "quality_label": "Rejected"
            }
        if brightness > self.MAX_BRIGHTNESS:
            print(f"❌ Rejected: Image overexposed (brightness: {brightness:.1f} > {self.MAX_BRIGHTNESS})")
            return None, {
                "error": "Image overexposed. Please reduce lighting.",
                "brightness": brightness,
                "quality_score": 0,
                "quality_label": "Rejected"
            }
        
        # Detect face regions
        region_colors, success, error, face_box = self.detect_face_regions(image)
        print(f"🔍 DEBUG: Face detection success = {success}, error = {error}")
        print(f"🔍 DEBUG: Regions found: {list(region_colors.keys()) if region_colors else 'None'}")
        
        if not success or not region_colors:
            return None, {"error": error or "No face regions detected", "quality_score": 0, "quality_label": "Rejected"}
        
        # Face area estimation
        if face_box:
            x, y, fw, fh = face_box
            face_area = fw * fh
            face_ratio = face_area / total_pixels
            print(f"🔍 DEBUG: Face ratio = {face_ratio:.3f}")
        else:
            face_ratio = 0.3
        
        if face_ratio < self.MIN_FACE_RATIO:
            print(f"❌ Rejected: Face too small (ratio: {face_ratio:.3f} < {self.MIN_FACE_RATIO})")
            return None, {
                "error": "Face too small. Please move closer to the camera.",
                "face_ratio": face_ratio,
                "quality_score": 0,
                "quality_label": "Rejected"
            }
        
        # Detect uneven lighting
        left_L = None
        right_L = None
        
        if "left_cheek" in region_colors:
            left_L = region_colors["left_cheek"][0]
        if "right_cheek" in region_colors:
            right_L = region_colors["right_cheek"][0]
        
        lighting_diff = 0
        lighting_warning = None
        
        if left_L is not None and right_L is not None:
            lighting_diff = abs(left_L - right_L)
            if lighting_diff > self.MAX_LIGHTING_DIFF:
                lighting_warning = f"Uneven lighting detected (diff: {lighting_diff:.1f})"
            print(f"🔍 DEBUG: Left cheek L* = {left_L:.1f}, Right cheek L* = {right_L:.1f}, Diff = {lighting_diff:.1f}")
        
        # Weighted average
        weighted_lab = np.zeros(3)
        total_weight = 0
        
        for region, lab in region_colors.items():
            weight = self.weights.get(region, 0)
            if weight > 0:
                weighted_lab += lab * weight
                total_weight += weight
        
        if total_weight > 0:
            avg_lab = weighted_lab / total_weight
        else:
            avg_lab = np.zeros(3)
        
        # Quality score
        quality_score = self._calculate_quality_score(brightness, lighting_diff, region_colors, face_ratio)
        quality_label = self._get_quality_label(quality_score)
        
        print(f"🔍 DEBUG: Quality score = {quality_score}/100 ({quality_label})")
        
        if quality_score < self.MIN_QUALITY_SCORE:
            return None, {
                "error": f"Image quality too low ({quality_score}/100). Please retake in natural daylight.",
                "quality_score": quality_score,
                "quality_label": quality_label,
                "brightness": brightness,
                "left_right_difference": lighting_diff,
                "lighting_warning": lighting_warning,
                "face_ratio": face_ratio
            }
        
        result = {
            "avg_lab": avg_lab.tolist(),
            "region_colors": {k: v.tolist() for k, v in region_colors.items()},
            "brightness": brightness,
            "left_right_difference": lighting_diff,
            "lighting_warning": lighting_warning,
            "quality_score": quality_score,
            "quality_label": quality_label,
            "face_ratio": face_ratio,
        }
        
        return avg_lab, result
    
    def _calculate_quality_score(self, brightness, lighting_diff, region_colors, face_ratio):
        """Calculate overall image quality score (0-100)"""
        score = 100
        
        if brightness < 80:
            score -= (80 - brightness) * 0.5
        elif brightness > 160:
            score -= (brightness - 160) * 0.5
        
        if lighting_diff > 15:
            score -= (lighting_diff - 15) * self.MAX_QUALITY_DIFF_WEIGHT
        
        if face_ratio < 0.20:
            score -= (0.20 - face_ratio) * 200
        
        required = ["left_cheek", "right_cheek"]
        for region in required:
            if region not in region_colors:
                score -= 15
        
        return max(0, min(100, round(score)))
    
    def _get_quality_label(self, score):
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Acceptable"
        else:
            return "Poor"
    
    def validate_image(self, image):
        """Quick validation before full processing"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        metrics = {
            "brightness": brightness,
            "width": w,
            "height": h,
        }
        
        if brightness < self.MIN_BRIGHTNESS:
            return False, "Image too dark. Please use better lighting.", metrics
        if brightness > self.MAX_BRIGHTNESS:
            return False, "Image overexposed. Please reduce lighting.", metrics
        
        return True, None, metrics
    
    def get_face_landmarks(self, image):
        """Get face bounding box for debugging"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return None
        
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return {"x": x, "y": y, "w": fw, "h": fh, "center_x": x + fw/2, "center_y": y + fh/2}
    
    # In app.py - UPGRADED IMPLEMENTATION

def get_combined_skin_color(img, sampling_points):
    """
    Combine manual pointers + automatic sampling for better accuracy.
    """
    h, w = img.shape[:2]
    all_colors = []
    
    # ── 1. MANUAL POINTERS (User-placed) ──
    if sampling_points and len(sampling_points) >= 2:
        for point in sampling_points:
            px = int(point['x'] * w)
            py = int(point['y'] * h)
            patch = get_patch(img, px, py)
            if patch is not None:
                avg_color = np.mean(patch, axis=(0, 1))
                all_colors.append(avg_color)
    
    # ── 2. AUTOMATIC SAMPLING (Face regions) ──
    detector = FaceDetector()
    region_colors, success, _ = detector.detect_face_regions(img)
    
    if success and region_colors:
        # Get colors from different regions
        left_cheek = region_colors.get("left_cheek")
        right_cheek = region_colors.get("right_cheek")
        jawline = region_colors.get("jawline_left")
        chin = region_colors.get("chin")
        
        if left_cheek is not None:
            all_colors.append(left_cheek)
        if right_cheek is not None:
            all_colors.append(right_cheek)
        if jawline is not None:
            all_colors.append(jawline)
        if chin is not None:
            all_colors.append(chin)
    
    # ── 3. WEIGHTED AVERAGE ──
    if len(all_colors) == 0:
        return np.mean(img, axis=(0, 1))
    
    # Give higher weight to manual pointers (user knows best)
    # Use the last 2 colors as pointers (if they exist)
    weights = []
    for i, color in enumerate(all_colors):
        if i < len(sampling_points) if sampling_points else 0:
            weights.append(0.25)  # Manual pointers get higher weight
        else:
            weights.append(0.15)  # Auto-sampled regions get lower weight
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    final_rgb = np.zeros(3)
    for i, color in enumerate(all_colors):
        final_rgb += color * weights[i]
    
    return final_rgb
