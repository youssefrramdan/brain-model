import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
MODEL_PATH = 'Brain_model_optimized.tflite'
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
INPUT_SIZE = 224

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("Model loaded successfully")

    # Get input quantization parameters if model is quantized
    input_scale = input_details[0]['quantization_parameters']['scale'] if input_details[0]['dtype'] == np.uint8 else None
    input_zero_point = input_details[0]['quantization_parameters']['zero_point'] if input_details[0]['dtype'] == np.uint8 else None
    is_quantized = input_details[0]['dtype'] == np.uint8

except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model input"""
    # Resize image
    image = image.resize((INPUT_SIZE, INPUT_SIZE))

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Quantize if model is quantized
    if is_quantized:
        img_array = img_array / input_scale + input_zero_point
        img_array = img_array.astype(np.uint8)

    return img_array

@app.route('/')
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Brain Tumor Classification API",
        "status": "active",
        "supported_classes": CLASS_NAMES
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict tumor class from MRI image
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        # Validate file type
        if not file.content_type.startswith('image/'):
            return jsonify({"error": "File must be an image"}), 400

        # Read and preprocess image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)

        # Make prediction
        pred_start = time.time()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)

        # Run inference
        interpreter.invoke()

        # Get prediction
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # If quantized, dequantize outputs
        if is_quantized:
            scale, zero_point = output_details[0]['quantization_parameters']['scale'], output_details[0]['quantization_parameters']['zero_point']
            predictions = (predictions.astype(np.float32) - zero_point) * scale

        # Get predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        # Calculate prediction time
        pred_time = time.time() - pred_start

        # Log prediction details
        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}, Time: {pred_time:.4f}s")

        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "predictions": {class_name: float(pred) for class_name, pred in zip(CLASS_NAMES, predictions[0])},
            "processing_time": pred_time
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "brain_tumor_classifier"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
