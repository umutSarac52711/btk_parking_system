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

# --- THE FIX: NEW PREPROCESSING FUNCTION ---
def preprocess_for_specialist_ocr(plate_image):
    """
    Prepares a cropped license plate for the specialist OCR model by
    resizing it to the exact dimensions the model requires (128x64).
    """
    # The model expects a fixed size of 128x64 pixels and a color image.
    target_size = (128, 64) # (Width, Height)
    
    # We resize the original color crop, ignoring the aspect ratio.
    resized_plate = cv2.resize(plate_image, target_size, interpolation=cv2.INTER_AREA)
    
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

        # We now use our new, correct preprocessing function
        preprocessed_for_ocr = preprocess_for_specialist_ocr(cropped_plate)
        
        if debug:
            debug_windows["1_Preprocessed_for_OCR"] = preprocessed_for_ocr

        # The OCR model now receives the exact input format it was trained on
        ocr_results = ocr_model.run(preprocessed_for_ocr)

        if ocr_results:
            plate_text = ocr_results[0]
            print(f"  -> Found: '{plate_text}'")
            detections.append({
                "text": plate_text,
                "confidence": yolo_confidence,
                "bbox": (x1, y1, x2, y2)
            })

    if not detections:
        print("YOLO found plates, but the specialist OCR could not read text.")
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
        print("\nPress any key in a window to close all.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return final_image, best_detection


# Main block for direct, interactive testing
if __name__ == "__main__":
    image_path = 'car_image_california.jpg' 
    final_image, result = recognize_plate(image_path, debug=True, display_windows=True)

    if result:
        print(f"\n--- SUCCESS ---")
        print(f"Recognized Plate Text: {result['text']}")
    else:
        print("\n--- FAILURE ---")