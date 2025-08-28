# yolo_approach.py
from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np

# (Configuration and helper functions remain the same)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
MODEL_PATH = 'license_plate_detector.pt'

def recognize_plate(image_path, super_debug=False):
    model = YOLO(MODEL_PATH)
    image = cv2.imread(image_path)
    if image is None: return None, None
    results = model.predict(image)
    detections = []

    for box in results[0].boxes:
        # ... (cropping, preprocessing, and clustering logic is the same and is perfect) ...
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        confidence = box.conf[0]
        cropped_plate = image[y1:y2, x1:x2]
        if cropped_plate.size == 0: continue
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        h, w = gray_plate.shape
        border_pixels = np.concatenate([gray_plate[0,:], gray_plate[h-1,:], gray_plate[:,0], gray_plate[:,w-1]])
        avg_brightness = np.mean(border_pixels)
        threshold_mode = cv2.THRESH_BINARY if avg_brightness > 127 else cv2.THRESH_BINARY_INV
        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, threshold_mode + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        dilated_plate = cv2.dilate(thresh_plate, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated_plate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potential_chars = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if h > 50 and w > 10: potential_chars.append({"x":x, "y":y, "w":w, "h":h})
        if not potential_chars: continue
        lines = {}
        for char in potential_chars:
            line_y = round(char['y'] / 50)
            if line_y not in lines: lines[line_y] = []
            lines[line_y].append(char)
        if not lines: continue
        best_line = max(lines.values(), key=len)
        best_line.sort(key=lambda c: c['x'])
        
        found_characters_ocr = []
        print("\n--- Tesseract OCR Details ---")
        
        # --- THE FINAL FIX: A SINGLE, ROBUST OCR CALL ---
        # We will re-assemble the line of characters into a single image strip
        # This gives Tesseract the context of a "word"
        
        # Get the bounding box of the entire line of characters
        min_char_x = min(c['x'] for c in best_line)
        min_char_y = min(c['y'] for c in best_line)
        max_char_x = max(c['x'] + c['w'] for c in best_line)
        max_char_y = max(c['y'] + c['h'] for c in best_line)

        # Crop the line of characters from the processed plate
        line_image = dilated_plate[min_char_y:max_char_y, min_char_x:max_char_x]
        
        # Invert and polish the entire line
        inverted_line = cv2.bitwise_not(line_image)
        polished_line = cv2.medianBlur(inverted_line, 3)
        padded_line = cv2.copyMakeBorder(polished_line, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
        
        if super_debug:
            cv2.imshow("Final Image Strip to Tesseract", padded_line)

        # The "Golden Configuration"
        custom_config = r'--oem 3 --psm 8 -c load_system_dawg=0 -c load_freq_dawg=0'
        
        ocr_text = pytesseract.image_to_string(padded_line, config=custom_config)
        
        cleaned_text = "".join(c for c in ocr_text if c.isalnum()).upper()
        
        print(f"  -> Tesseract read full line as: '{cleaned_text}'")
        
        if cleaned_text:
            detections.append({ "text": cleaned_text, "confidence": confidence, "bbox": (x1, y1, x2, y2) })

    if not detections: return image, None
    best_detection = max(detections, key=lambda d: d['confidence'])
    
    # ... (Final display logic is the same) ...
    final_image = image.copy()
    b = best_detection['bbox']
    cv2.rectangle(final_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
    cv2.putText(final_image, best_detection['text'], (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    return final_image, best_detection

# Main block for direct testing
if __name__ == "__main__":
    final_image, result = recognize_plate('car_image_california.jpg', super_debug=True)

    if result:
        print(f"\n--- SUCCESS ---")
        print(f"Recognized Plate Text: {result['text']}")
        cv2.imshow("Final Result", final_image)
    else:
        print("\n--- FAILURE ---")
        
    print("\nPress any key to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()