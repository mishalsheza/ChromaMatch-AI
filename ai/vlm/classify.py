import os
import logging
import json
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global client instance
_client = None

def get_client():
    """Get or create the VLM client"""
    global _client
    if _client is None:
        provider = os.environ.get("VLM_PROVIDER", "mock")
        if provider == "gguf":
            try:
                from .gguf_adapter import MiniCPMGGUF
                _client = MiniCPMGGUF()
            except ImportError as e:
                logger.warning(f"GGUF adapter not found: {e}")
                _client = MockVLMClient()
        else:
            logger.info(f"Using {provider} VLM provider")
            if provider == "mock":
                _client = MockVLMClient()
            else:
                logger.warning(f"Unknown provider {provider}, using mock")
                _client = MockVLMClient()
    return _client

class MockVLMClient:
    """Mock client for testing without model weights"""
    def classify_undertone(self, image_array):
        # Simple rule-based fallback based on image statistics
        try:
            if len(image_array.shape) == 3:
                avg_r = np.mean(image_array[:,:,0])
                avg_g = np.mean(image_array[:,:,1])
                avg_b = np.mean(image_array[:,:,2])
                
                # Very simple heuristic
                if avg_r > avg_g and avg_r > avg_b:
                    undertone = "warm"
                elif avg_b > avg_r and avg_b > avg_g:
                    undertone = "cool"
                else:
                    undertone = "neutral"
                
                # Estimate complexion
                brightness = (avg_r + avg_g + avg_b) / 3
                if brightness > 200:
                    complexion = "very_light"
                elif brightness > 170:
                    complexion = "light"
                elif brightness > 130:
                    complexion = "medium"
                elif brightness > 90:
                    complexion = "tan"
                else:
                    complexion = "deep"
                
                return {
                    "undertone": undertone,
                    "complexion": complexion,
                    "confidence": 0.6,
                    "reasoning": "Mock classification based on image statistics"
                }
        except Exception as e:
            logger.warning(f"Mock classification error: {e}")
        
        return {
            "undertone": "neutral",
            "complexion": "medium",
            "confidence": 0.5,
            "reasoning": "Default mock response"
        }
    
    def generate_text(self, prompt: str, image_array: Optional[np.ndarray] = None) -> str:
        """Generate text response (mock implementation)"""
        return f"Mock response for: {prompt[:50]}..."

def classify_undertone(image_array):
    """Classify skin undertone from image array"""
    client = get_client()
    return client.classify_undertone(image_array)

def generate_text(prompt: str, image_array: Optional[np.ndarray] = None) -> str:
    """Generate text using the VLM (for recommendation generation)"""
    client = get_client()
    if hasattr(client, 'generate_text'):
        return client.generate_text(prompt, image_array)
    else:
        # Fallback for clients without generate_text
        return f"Generated text for: {prompt[:50]}..."
