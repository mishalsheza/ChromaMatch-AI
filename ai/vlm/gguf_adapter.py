import os
import json
import base64
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class MiniCPMGGUF:
    """Adapter for MiniCPM-V-2.6 GGUF model using llama-cpp-python"""
    
    def __init__(self):
        self.model_path = os.environ.get(
            "VLM_MODEL_PATH", 
            "models/minicpm-v-2.6-Q4_K_M.gguf"
        )
        self.loaded = False
        self.use_metal = os.environ.get("USE_METAL", "true").lower() == "true"
        self._load_model()
    
    def _load_model(self):
        try:
            from llama_cpp import Llama
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}")
                logger.info("Download from: https://huggingface.co/freakyX0/MiniCPM-V-2.6-Arm-AAArch64-GGUF")
                self.loaded = False
                return
            
            # Load model with Metal support for Apple Silicon
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=-1 if self.use_metal else 0,  # -1 = use all GPU layers
                verbose=False
            )
            self.loaded = True
            logger.info(f"✅ MiniCPM GGUF loaded from {self.model_path}")
            logger.info(f"✅ GPU acceleration: {self.use_metal}")
            
        except ImportError:
            logger.warning("❌ llama-cpp-python not installed")
            logger.info("Install with: pip install llama-cpp-python")
            self.loaded = False
        except Exception as e:
            logger.warning(f"❌ Failed to load GGUF model: {e}")
            self.loaded = False
    
    def classify_undertone(self, image_array):
        """Classify skin undertone from image array"""
        if not self.loaded:
            return self._mock_response("Model not loaded")
        
        try:
            # Convert numpy array to base64 image
            # Ensure image is in correct format
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            img = Image.fromarray(image_array.astype('uint8'))
            
            # Resize to reasonable size for faster processing
            img.thumbnail((448, 448))
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Build the prompt
            prompt = """
            You are a professional skin tone analyst. Analyze the skin tone in this image and return ONLY valid JSON.

            Rules:
            1. Undertone must be one of: "warm", "cool", "neutral", "olive"
            2. Complexion must be one of: "very_light", "light", "medium", "tan", "deep"
            3. Confidence must be between 0 and 1
            4. Provide brief reasoning

            Example response:
            {"undertone": "warm", "complexion": "medium", "confidence": 0.85, "reasoning": "Golden undertones visible in the skin"}
            
            Return JSON only, no other text.
            """
            
            # Call the model
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a skin tone analyst. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.1,
                stop=["}"]  # Stop after JSON completes
            )
            
            # Parse the response
            result_text = response['choices'][0]['message']['content']
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)
            
            # Validate and return
            return {
                "undertone": result.get("undertone", "neutral"),
                "complexion": result.get("complexion", "medium"),
                "confidence": min(1.0, max(0.0, result.get("confidence", 0.5))),
                "reasoning": result.get("reasoning", "GGUF model classification")
            }
            
        except Exception as e:
            logger.warning(f"⚠️ GGUF inference error: {e}")
            return self._mock_response(f"Inference failed: {str(e)[:50]}")
    
    def _mock_response(self, reason="Model unavailable"):
        return {
            "undertone": "warm",
            "complexion": "medium",
            "confidence": 0.5,
            "reasoning": f"Fallback: {reason}"
        }