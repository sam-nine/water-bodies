from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import base64
import io
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime
import logging

# Import your inference service
# Make sure this path is correct for your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.inference_service import WaterSegmentationInference

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference service instance
inference_service = None

def initialize_model():
    """Initialize the water segmentation model"""
    global inference_service
    try:
        model_uri = "runs:/d792473199e2474d8ba76b89ef32c62e/pytorch_model"
        
        inference_service = WaterSegmentationInference(
            mlflow_model_uri=model_uri,
            confidence_threshold=0.5  # Default threshold
        )
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

def numpy_to_base64(image_array):
    """Convert numpy array to base64 string"""
    if len(image_array.shape) == 2:  # Grayscale
        # Convert to 0-255 range if needed
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_array, mode='L')
    else:  # RGB
        image = Image.fromarray(image_array.astype(np.uint8))
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference_service is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Process uploaded images and return segmentation results"""
    try:
        if inference_service is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Get threshold from request
        threshold = float(request.form.get('threshold', 0.5))
        
        # Update inference service threshold
        inference_service.confidence_threshold = threshold
        
        # Get uploaded files
        files = request.files.getlist('images')
        
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        
        for file in files:
            try:
                # Read image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Process image
                result = inference_service.predict_single_image(image)
                
                # Convert original image to base64
                original_base64 = numpy_to_base64(np.array(image))
                
                # Convert masks to base64
                binary_mask_base64 = numpy_to_base64(result['binary_mask'])
                prob_mask_base64 = numpy_to_base64(result['probability_mask'])
                
                # Create overlay
                overlay = create_overlay(np.array(image), result['binary_mask'])
                overlay_base64 = numpy_to_base64(overlay)
                
                # Prepare result
                file_result = {
                    'filename': file.filename,
                    'originalImage': original_base64,
                    'maskImage': binary_mask_base64,
                    'probabilityMask': prob_mask_base64,
                    'overlayImage': overlay_base64,
                    'waterPercentage': round(result['water_percentage'], 2),
                    'confidence': round(result['average_confidence'], 3),
                    'maxConfidence': round(result['max_confidence'], 3),
                    'waterPixels': int(result['water_pixels']),
                    'totalPixels': int(result['total_pixels']),
                    'threshold': threshold,
                    'timestamp': result['timestamp']
                }
                
                results.append(file_result)
                logger.info(f"Processed {file.filename}: {result['water_percentage']:.2f}% water")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

def create_overlay(original_image, binary_mask):
    """Create overlay of original image with water mask"""
    overlay = original_image.copy()
    
    # Ensure mask is in correct format
    if binary_mask.max() <= 1.0:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    
    # Resize mask to match original image if needed
    if original_image.shape[:2] != binary_mask.shape:
        binary_mask = cv2.resize(binary_mask, 
                                (original_image.shape[1], original_image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Create blue overlay for water areas
    water_overlay = np.zeros_like(original_image)
    water_mask = binary_mask > 128  # Threshold for binary mask
    water_overlay[water_mask] = [0, 100, 255]  # Blue color
    
    # Blend with original image
    overlay = cv2.addWeighted(overlay, 0.7, water_overlay, 0.3, 0)
    
    return overlay

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    """Update the confidence threshold"""
    try:
        data = request.get_json()
        threshold = float(data.get('threshold', 0.5))
        
        if inference_service:
            inference_service.confidence_threshold = threshold
            return jsonify({'success': True, 'threshold': threshold})
        else:
            return jsonify({'error': 'Model not initialized'}), 500
            
    except Exception as e:
        logger.error(f"Error updating threshold: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize model on startup
    model_loaded = initialize_model()
    
    if not model_loaded:
        logger.warning("Model initialization failed. Some features may not work.")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)