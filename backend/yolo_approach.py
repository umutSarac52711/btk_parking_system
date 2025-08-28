# yolo_approach.py
from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np

# (Configuration and helper functions remain the same)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
MODEL_PATH = 'license_plate_detector.pt'

def recognize_plate_yolo(image_path, debug=True, display_windows=True):
    # ... (initial YOLO setup is the same) ...
    model = YOLO(MODEL_PATH)
    image = cv2.imread(image_path)
    if image is None: return None, None
    results = model.predict(image)
    detections = []
    debug_windows = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        confidence = box.conf[0]
        cropped_plate = image[y1:y2, x1:x2]
        if cropped_plate.size == 0: continue

        # (Preprocessing and clustering logic is the same and is working perfectly)
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        h, w = gray_plate.shape
        border_pixels = np.concatenate([gray_plate[0,:], gray_plate[h-1,:], gray_plate[:,0], gray_plate[:,w-1]])
        avg_brightness = np.mean(border_pixels)
        threshold_mode = cv2.THRESH_BINARY if avg_brightness > 127 else cv2.THRESH_BINARY_INV
        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, threshold_mode + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        dilated_plate = cv2.dilate(thresh_plate, kernel, iterations=1)
        if debug: debug_windows.append(("1 - Preprocessed & Dilated", dilated_plate))
        contours, _ = cv2.findContours(dilated_plate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_debug_image = cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2BGR)
        
        potential_chars = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 50 and w > 10: potential_chars.append({"x": x, "y": y, "w": w, "h": h})
        if not potential_chars: continue
        lines = {}
        for char in potential_chars:
            line_y = round(char['y'] / 50)
            if line_y not in lines: lines[line_y] = []
            lines[line_y].append(char)
        best_line = max(lines.values(), key=len)
        
        # --- FIX #1: SORT THE CHARACTERS BEFORE OCR/DEBUGGING ---
        best_line.sort(key=lambda c: c['x'])

        for char in best_line: cv2.rectangle(contour_debug_image, (char['x'], char['y']), (char['x']+char['w'], char['y']+char['h']), (0, 255, 0), 2)
        if debug: debug_windows.append(("2 - Final Character Line (Green)", contour_debug_image))
        
        found_characters_ocr = []
        for i, char_data in enumerate(best_line):
            x, y, w, h = char_data['x'], char_data['y'], char_data['w'], char_data['h']
            char_image = dilated_plate[y:y+h, x:x+w]
            inverted_char = cv2.bitwise_not(char_image)
            polished_char = cv2.medianBlur(inverted_char, 3)
            padded_char = cv2.copyMakeBorder(polished_char, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
            if debug: debug_windows.append((f"Char #{i+1}", padded_char))
            
            # --- FIX #2: USE THE TESSERACT LEGACY ENGINE (OEM 0) ---
            custom_config = r'--oem 0 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 10'
            
            char_text = pytesseract.image_to_string(padded_char, config=custom_config)
            cleaned_char = "".join(c for c in char_text if c.isalnum()).upper()
            if cleaned_char:
                found_characters_ocr.append(cleaned_char)

        if found_characters_ocr:
            plate_text = "".join(found_characters_ocr) # No need to sort again
            detections.append({ "text": plate_text, "confidence": confidence, "bbox": (x1, y1, x2, y2) })

    # (Display and final selection logic is the same)
    if debug and display_windows:
        print("\nDisplaying debug windows...")
        for name, img in debug_windows:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, img)

    if not detections:
        print("\nYOLO found plates, but segmentation/OCR failed.")
        # CHANGE THIS RETURN STATEMENT:
        return image, None

    best_detection = max(detections, key=lambda d: d['confidence'])
    
    final_image = image.copy()
    b = best_detection['bbox']
    cv2.rectangle(final_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
    cv2.putText(final_image, best_detection['text'], (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    if display_windows:
        cv2.imshow("Final Result", final_image)
        print("\nPress any key to close all windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # CHANGE THIS FINAL RETURN STATEMENT:
    return final_image, best_detection

def main():
    image_path = 'car_image_california.jpg'
    final_image, plate_text = recognize_plate_yolo(image_path, debug=True)

    if plate_text:
        print(f"\n--- SUCCESS ---")
        print(f"Recognized Plate Text: {plate_text}")
        cv2.imshow("Final Result", final_image)
    else:
        print("\n--- FAILURE ---")
        
    print("\nPress any key to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()