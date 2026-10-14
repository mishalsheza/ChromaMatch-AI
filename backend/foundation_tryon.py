"""
foundation_tryon.py — rough virtual try-on preview (experimental).

The target colour is the shade's hex code from ai/data/foundations.json, the same
database the recommender matches against, so "Try On" shows the shade that was
actually recommended.

Method: find skin (HSV + YCrCb colour ranges, plus an oval around the Haar face box),
then shift the masked pixels' colour towards the shade by `intensity`. This is
simple alpha blending, not texture-aware recolouring, so treat it as a preview.
"""

import base64

import cv2
import numpy as np

# Fraction of the shift actually applied at intensity=1.0. Full replacement looks
# painted-on; partial blending keeps some of the skin's own variation.
BLEND_STRENGTH = 0.5


class FoundationTryOn:
    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def apply_foundation(self, img: np.ndarray, target_rgb, intensity: float = 0.6) -> str:
        """
        Args:
            img: BGR image array (decoded in memory; uploads are never written to disk)
            target_rgb: (R, G, B) of the foundation shade, 0–255
            intensity: 0–1, how strongly the shade is applied

        Returns:
            base64-encoded JPEG (re-encoded by OpenCV, so no EXIF metadata survives)
        """
        if img is None:
            raise ValueError("Could not load image")
        intensity = float(np.clip(intensity, 0.0, 1.0))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.maximum(self._detect_skin(img_rgb), self._detect_face_mask(img_rgb))
        result = self._apply_color_shift(img_rgb, mask, np.array(target_rgb, dtype=np.float32), intensity)
        return self._image_to_base64(result)

    @staticmethod
    def _detect_skin(image: np.ndarray) -> np.ndarray:
        """Soft skin mask from common HSV and YCrCb skin-colour ranges."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        mask_hsv = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([25, 170, 255]))
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        combined = cv2.bitwise_or(mask_hsv, mask_ycrcb)

        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.GaussianBlur(combined, (21, 21), 0)
        return combined / 255.0

    def _detect_face_mask(self, image: np.ndarray) -> np.ndarray:
        """Soft oval mask around each detected face, widened to include the cheeks."""
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        if self._face_cascade.empty():   # cascade file missing: rely on the colour mask only
            return mask
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            centre = (int(x + w // 2), int(y + h // 2))
            cv2.ellipse(mask, centre, (int(w * 0.6), int(h * 0.75)), 0, 0, 360, 1, -1)
        return cv2.GaussianBlur(mask, (31, 31), 0)

    @staticmethod
    def _apply_color_shift(image: np.ndarray, mask: np.ndarray, target: np.ndarray, intensity: float) -> np.ndarray:
        """Move masked pixels' colour towards `target`, keeping their light/dark variation."""
        img = image.astype(np.float32)
        skin = img[mask > 0.3]
        avg_skin = skin.mean(axis=0) if len(skin) else img.reshape(-1, 3).mean(axis=0)
        shift = target - avg_skin  # same offset for every pixel → texture and shading are kept
        weight = (mask * intensity * BLEND_STRENGTH)[..., None]
        return np.clip(img + shift * weight, 0, 255).astype(np.uint8)

    @staticmethod
    def _image_to_base64(image_rgb: np.ndarray) -> str:
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Could not encode result image")
        return base64.b64encode(buffer).decode("ascii")
