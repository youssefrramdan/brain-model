from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging
from pathlib import Path

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Brain Tumor Classification API",
    description="API for classifying brain tumors using MRI images",
    version="1.0.0"
)

# التأكد من وجود النموذج
MODEL_PATH = "Brain_model.tflite"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please ensure the model file is in the correct location.")

try:
    # تحميل نموذج TFLite
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # الحصول على معلومات المدخلات والمخرجات
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# أسماء الفئات
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    تجهيز الصورة للتصنيف
    """
    try:
        img = img.resize((224, 224))  # VGG16 input size
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def get_prediction(image_array: np.ndarray) -> np.ndarray:
    """
    الحصول على التنبؤات باستخدام نموذج TFLite
    """
    try:
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

@app.get("/")
async def root():
    """
    نقطة النهاية الرئيسية للتحقق من حالة API
    """
    return {
        "status": "active",
        "model": "Brain Tumor Classification (TFLite)",
        "supported_classes": CLASS_NAMES,
        "input_shape": input_details[0]['shape'].tolist()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    تصنيف صورة الرنين المغناطيسي للدماغ
    """
    # التحقق من نوع الملف
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # تجهيز الصورة
        img_array = preprocess_image(img)

        # التنبؤ
        predictions = get_prediction(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
