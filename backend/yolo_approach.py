# backend/yolo_approach.py
from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np

# (Configuration is the same)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
MODEL_PATH = 'license_plate_detector.pt'

# We'll rename the function to make its purpose clearer
def recognize_plate(image_path, super_debug=False):
    model = YOLO(MODEL_PATH)
    image = cv2.imread(image_path)
    if image is None: return None, None

    results = model.predict(image)
    detections = []
    debug_windows = []

    # --- SUPER DEBUG: VISUALIZE ALL YOLO DETECTIONS ---
    if super_debug:
        yolo_debug_image = image.copy()
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            conf = box.conf[0]
            cv2.rectangle(yolo_debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(yolo_debug_image, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        debug_windows.append(("1 - All YOLO Detections", yolo_debug_image))

    # We now process every plate YOLO finds
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        confidence = box.conf[0]
        cropped_plate = image[y1:y2, x1:x2]
        if cropped_plate.size == 0: continue

        # (Preprocessing and clustering logic is the same)
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
        
        # --- SUPER DEBUG: VISUALIZE SEGMENTATION ---
        if super_debug:
            contour_debug_image = cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2BGR)
            for char in best_line:
                cv2.rectangle(contour_debug_image, (char['x'], char['y']), (char['x']+char['w'], char['y']+char['h']), (0, 255, 0), 2)
            debug_windows.append(("2 - Segmented Characters", contour_debug_image))

        found_characters_ocr = []
        print("\n--- Tesseract OCR Details ---")
        for i, char_data in enumerate(best_line):
            print(f"Processing Char #{i+1}: Bounding Box {char_data}")
            x, y, w, h = char_data['x'], char_data['y'], char_data['w'], char_data['h']
            char_image = dilated_plate[y:y+h, x:x+w]
            inverted_char = cv2.bitwise_not(char_image)
            polished_char = cv2.medianBlur(inverted_char, 3)
            padded_char = cv2.copyMakeBorder(polished_char, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
            
            # --- THE FIX #1: CHANGE TESSERACT'S MODE ---
            # PSM 7 treats the image as a single line of text, which is more robust.
            custom_config = r'--oem 0 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7'
            
            ocr_data = pytesseract.image_to_data(padded_char, config=custom_config, output_type=pytesseract.Output.DICT)

            print(f"  Char #{i+1}: OCR Data: {ocr_data}")

            # Find the character with the highest confidence in the snippet
            if ocr_data['text']:
                highest_conf_idx = -1
                max_conf = -1
                for j, text in enumerate(ocr_data['text']):
                    print(f"    OCR Candidate: '{text}' with confidence {ocr_data['conf'][j]}")
                    if text.strip() and int(ocr_data['conf'][j]) > max_conf:
                        max_conf = int(ocr_data['conf'][j])
                        highest_conf_idx = j
                        print(f"    -> New highest confidence candidate: '{text}' with confidence {max_conf}")
                
                if highest_conf_idx != -1:
                    char_text = ocr_data['text'][highest_conf_idx]
                    char_conf = ocr_data['conf'][highest_conf_idx]
                    print(f"  Char #{i+1}: Read '{char_text}' with confidence {char_conf}")
                    found_characters_ocr.append(char_text)
                    found_char = True
                    print(f"  Char #{i+1}: -> TESSERACT READ SUCCESSFULLY")

            # --- THE FIX #2: MAKE THE SILENT FAILURE LOUD ---
            if not found_char:
                print(f"  Char #{i+1}: -> TESSERACT FAILED TO READ")

        if found_characters_ocr:
            plate_text = "".join(found_characters_ocr)
            detections.append({ "text": plate_text, "confidence": confidence, "bbox": (x1, y1, x2, y2) })

    if not detections: return image, None

    best_detection = max(detections, key=lambda d: d['confidence'])
    
    # Final visualization logic
    final_image = image.copy()
    b = best_detection['bbox']
    cv2.rectangle(final_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
    cv2.putText(final_image, best_detection['text'], (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if super_debug:
        print("\nDisplaying all debug windows...")
        debug_windows.append(("3 - Final Result", final_image))
        for name, img in debug_windows:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, img)
        print("\nPress any key in a window to close all.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_image, best_detection

def main():
    # Use the super_debug flag to get maximum visibility
    recognize_plate('car_image_california.jpg', super_debug=True)

if __name__ == "__main__":
    main()