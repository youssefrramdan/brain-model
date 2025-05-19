import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load the original model
model = load_model('Brain_model.keras')

# Define the optimization parameters
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# Enable experimental features for further size reduction
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Representative dataset generator
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3) * 255
        yield [data.astype(np.float32)]

converter.representative_dataset = representative_dataset

# Convert the model
tflite_model = converter.convert()

# Save the quantized model
with open('Brain_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

# Print size comparison
import os
original_size = os.path.getsize('Brain_model.keras') / (1024 * 1024)
new_size = os.path.getsize('Brain_model_optimized.tflite') / (1024 * 1024)
print(f"Original model size: {original_size:.2f} MB")
print(f"Optimized model size: {new_size:.2f} MB")
print(f"Size reduction: {((original_size - new_size) / original_size * 100):.2f}%")
