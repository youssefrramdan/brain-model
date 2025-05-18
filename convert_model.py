import tensorflow as tf

# 1. تحميل النموذج الأصلي
model = tf.keras.models.load_model("Brain_model.keras")

# 2. التحويل إلى TFLite (بدون ضغط)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. حفظ النموذج الجديد
with open("Brain_model.tflite", "wb") as f:
    f.write(tflite_model)

# 4. طباعة فرق الحجم
import os

original_size = os.path.getsize("Brain_model.keras") / 1024 / 1024
tflite_size = os.path.getsize("Brain_model.tflite") / 1024 / 1024

print(f"الحجم الأصلي: {original_size:.2f} MB")
print(f"حجم TFLite: {tflite_size:.2f} MB")
