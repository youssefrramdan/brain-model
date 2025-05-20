import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from glob import glob

def preprocess_image(image_path):
    """Preprocess a single image for model input"""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array[np.newaxis, ...]

def get_representative_dataset():
    """Get representative dataset from test images"""
    test_images = glob('test_images/*/*.jpg') + glob('test_images/*/*.jpeg') + glob('test_images/*/*.png')
    if not test_images:
        raise ValueError("No test images found! Please add some test images to test_images directory")

    def representative_dataset():
        for image_path in test_images:
            img = preprocess_image(image_path)
            yield [img.astype(np.float32)]

    return representative_dataset

def convert_to_tflite():
    print("Loading original model...")
    model = load_model('Brain_model.keras')

    print("Creating TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Configure quantization
    converter.target_spec.supported_types = [tf.float16]  # Using float16 for better accuracy
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Use actual representative dataset
    try:
        converter.representative_dataset = get_representative_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        print("Using INT8 quantization with representative dataset")
    except ValueError as e:
        print(f"Warning: {e}")
        print("Falling back to float16 quantization without representative dataset")

    print("Converting model...")
    tflite_model = converter.convert()

    # Save the model
    with open('Brain_model_optimized.tflite', 'wb') as f:
        f.write(tflite_model)

    # Print size comparison
    original_size = os.path.getsize('Brain_model.keras') / (1024 * 1024)
    new_size = os.path.getsize('Brain_model_optimized.tflite') / (1024 * 1024)
    print(f"\nOriginal model size: {original_size:.2f} MB")
    print(f"Optimized model size: {new_size:.2f} MB")
    print(f"Size reduction: {((original_size - new_size) / original_size * 100):.2f}%")

if __name__ == "__main__":
    convert_to_tflite()
