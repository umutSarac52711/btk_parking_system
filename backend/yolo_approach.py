# backend/yolo_approach.py
from ultralytics import YOLO
import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer

# --- Configuration ---
MODEL_PATH = 'license_plate_detector.pt'

# --- Initialize Models ---
try:
    yolo_model = YOLO(MODEL_PATH)
    ocr_model = LicensePlateRecognizer('cct-xs-v1-global-model')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    yolo_model = None
    ocr_model = None

# --- THE FINAL, BEST PREPROCESSING FUNCTION ---
def preprocess_for_specialist_ocr(plate_image):
    """
    Prepares a plate by enhancing contrast with CLAHE and resizing
    to the exact format required by the specialist OCR model.
    """
    # 1. Convert to grayscale to perform contrast enhancement
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # 3. Convert the enhanced grayscale back to a 3-channel BGR image
    # The model expects 3 channels, even if they are identical.
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    # 4. Resize to the model's required input size (128x64)
    target_size = (128, 64) # (Width, Height)
    resized_plate = cv2.resize(enhanced_bgr, target_size, interpolation=cv2.INTER_AREA)
    
    return resized_plate

def recognize_plate(image_path, debug=False, display_windows=False):
    if not yolo_model or not ocr_model:
        return cv2.imread(image_path), None

    image = cv2.imread(image_path)
    if image is None: return None, None

    yolo_results = yolo_model.predict(image, verbose=False)
    detections = []
    debug_windows = {}

    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        yolo_confidence = box.conf[0]
        
        cropped_plate = image[y1:y2, x1:x2]
        if cropped_plate.size == 0: continue

        # Use our new, advanced preprocessing function
        preprocessed_for_ocr = preprocess_for_specialist_ocr(cropped_plate)
        
        if debug:
            debug_windows["1_Preprocessed_for_OCR"] = preprocessed_for_ocr

        ocr_results = ocr_model.run(preprocessed_for_ocr)

        if ocr_results:
            # Clean the output to remove padding characters like '_'
            raw_text = ocr_results[0]
            plate_text = "".join(filter(str.isalnum, raw_text))
            
            print(f"  -> Raw OCR: '{raw_text}' -> Cleaned: '{plate_text}'")
            detections.append({
                "text": plate_text,
                "confidence": yolo_confidence,
                "bbox": (x1, y1, x2, y2)
            })

    if not detections:
        return image, None

    best_detection = max(detections, key=lambda d: d['confidence'])
    
    final_image = image.copy()
    b = best_detection['bbox']
    cv2.rectangle(final_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
    cv2.putText(final_image, best_detection['text'], (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    if display_windows:
        if debug:
            for name, img in debug_windows.items():
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.imshow(name, img)
        cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Result", final_image)
        print("\nPress any key to close all windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return final_image, best_detection


# Main block for direct, interactive testing
if __name__ == "__main__":
    # Test with both challenging images
    print("--- Testing California Plate ---")
    recognize_plate('car_image_california.jpg', debug=True, display_windows=True)
    
    print("\n\n--- Testing German Plate ---")
    recognize_plate('dataset/car license plate close up/image_13.jpg', debug=True, display_windows=True)