"""
Test your trained AI model
"""
import tensorflow as tf
import cv2
import numpy as np
import os

# Load model
model = tf.keras.models.load_model('ai/models/balanced_skin_analyzer.h5')
print("✅ Model loaded!")

# Test on a random image from your dataset
test_folder = 'ai/data/raw/light_cool'  # Change this to test different categories
test_images = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png'))]

if test_images:
    test_img_path = os.path.join(test_folder, test_images[0])
    img = cv2.imread(test_img_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    skin_pred, under_pred = model.predict(img)
    
    skin_labels = ['Light', 'Medium', 'Deep']
    under_labels = ['Cool', 'Warm', 'Neutral', 'Olive']
    
    print(f"\n📸 Test Image: {test_images[0]}")
    print(f"🎨 Predicted Skin Tone: {skin_labels[np.argmax(skin_pred[0])]} ({np.max(skin_pred[0])*100:.1f}%)")
    print(f"🎨 Predicted Undertone: {under_labels[np.argmax(under_pred[0])]} ({np.max(under_pred[0])*100:.1f}%)")