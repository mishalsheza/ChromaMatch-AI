"""
CNN Model for skin tone and undertone classification
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications

class SkinAnalyzerModel:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.skin_tone_labels = ['Light', 'Medium', 'Deep']
        self.undertone_labels = ['Cool', 'Warm', 'Neutral', 'Olive']
    
    def build_model(self):
        """Build transfer learning model using MobileNetV2"""
        # Load pre-trained MobileNetV2
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = False
        
        # Build custom head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-output layers
        skin_tone = layers.Dense(3, activation='softmax', name='skin_tone')(x)
        undertone = layers.Dense(4, activation='softmax', name='undertone')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=[skin_tone, undertone])
        return model
    
    def compile_model(self, model):
        """Compile with appropriate settings"""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'skin_tone': 'categorical_crossentropy',
                'undertone': 'categorical_crossentropy'
            },
            metrics={
                'skin_tone': 'accuracy',
                'undertone': 'accuracy'
            }
        )
        return model