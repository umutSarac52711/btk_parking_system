# backend/services/recognition_service.py
from ultralytics import YOLO
import cv2
from fast_plate_ocr import LicensePlateRecognizer

# --- Configuration ---
YOLO_MODEL_PATH = 'license_plate_detector.pt'

# --- Model Caching ---
# Load models once when the service is initialized for maximum efficiency
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    ocr_model = LicensePlateRecognizer('cct-xs-v1-global-model')
    print("Recognition models loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading recognition models: {e}")
    yolo_model = None
    ocr_model = None

def _preprocess_for_ocr(plate_image):
    """Prepares a plate by enhancing contrast and resizing."""
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    target_size = (128, 64)
    resized_plate = cv2.resize(enhanced_bgr, target_size, interpolation=cv2.INTER_AREA)
    return resized_plate

def recognize_plate_from_image(image_bgr):
    """
    Takes a raw BGR image (from OpenCV) and returns a list of recognized plates.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              {'text': str, 'confidence': float, 'bbox': tuple}
    """
    if not yolo_model or not ocr_model:
        return []

    yolo_results = yolo_model.predict(image_bgr, verbose=False)
    detections = []

    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        yolo_confidence = float(box.conf[0])
        
        cropped_plate = image_bgr[y1:y2, x1:x2]
        if cropped_plate.size == 0:
            continue

        preprocessed_crop = _preprocess_for_ocr(cropped_plate)
        ocr_results = ocr_model.run(preprocessed_crop)

        if ocr_results:
            raw_text = ocr_results[0]
            plate_text = "".join(filter(str.isalnum, raw_text))
            
            detections.append({
                "text": plate_text,
                "confidence": yolo_confidence,
                "bbox": (x1, y1, x2, y2)
            })
            
    return detections