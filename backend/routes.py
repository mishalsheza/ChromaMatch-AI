from flask import Blueprint, request, jsonify
import cv2
import numpy as np

api_bp = Blueprint("api", __name__)

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@api_bp.route("/analyze", methods=["POST"])
def analyze():
    """Redirect to AI endpoint or provide message"""
    return jsonify({
        "success": False,
        "message": "Please use POST /api/ai/analyze for skin analysis"
    }), 200