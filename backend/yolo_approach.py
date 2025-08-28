# backend/yolo_approach.py
from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
MODEL_PATH = 'license_plate_detector.pt'


def recognize_plate(image_path, debug=False, display_windows=False):
    """
    Detects and recognizes a license plate from an image using a YOLOv8 model for detection
    and a Tesseract-based character segmentation pipeline for OCR.

    Args:
        image_path (str): Path to the input image.
        debug (bool): If True, generates data for debug visualizations.
        display_windows (bool): If True, displays the debug and final result windows.

    Returns:
        tuple: A tuple containing the final annotated image and the best detection result dictionary.
               Returns (original_image, None) if no plate is successfully recognized.
    """
    model = YOLO(MODEL_PATH)
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    results = model.predict(image)
    detections = []
    debug_windows = {}

    # Process each bounding box detected by YOLO
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        yolo_confidence = box.conf[0]
        cropped_plate = image[y1:y2, x1:x2]
        if cropped_plate.size == 0:
            continue

        # --- 1. Preprocessing for Contour Detection ---
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

        # Smartly decide whether to invert the threshold based on border brightness
        h, w = gray_plate.shape
        border_pixels = np.concatenate([gray_plate[0, :], gray_plate[h - 1, :], gray_plate[:, 0], gray_plate[:, w - 1]])
        avg_brightness = np.mean(border_pixels)
        threshold_mode = cv2.THRESH_BINARY if avg_brightness > 127 else cv2.THRESH_BINARY_INV

        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, threshold_mode + cv2.THRESH_OTSU)
        
        # Dilate to connect fragmented parts of characters
        kernel = np.ones((3, 3), np.uint8)
        dilated_plate = cv2.dilate(thresh_plate, kernel, iterations=1)

        if debug:
            debug_windows["1_Preprocessed"] = dilated_plate

        # --- 2. Character Segmentation and Clustering ---
        contours, _ = cv2.findContours(dilated_plate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        potential_chars = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Initial loose filter for character-like shapes
            # if h > 30 and 10 < w < 200:
            #     potential_chars.append({"x": x, "y": y, "w": w, "h": h})

            if h > 50 and w > 10: potential_chars.append({"x":x, "y":y, "w":w, "h":h})
        
        if not potential_chars:
            continue

        # Cluster characters into lines based on their vertical position
        lines = {}
        for char in potential_chars:
            line_y = round(char['y'] / 50)  # Group in bands of 50 pixels
            if line_y not in lines:
                lines[line_y] = []
            lines[line_y].append(char)
        
        if not lines:
            continue

        # Assume the main plate number is the line with the most characters
        best_line = max(lines.values(), key=len)
        best_line.sort(key=lambda c: c['x']) # Sort characters left-to-right

        if debug:
            contour_debug_image = cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2BGR)
            for char in best_line:
                cv2.rectangle(contour_debug_image, (char['x'], char['y']), (char['x'] + char['w'], char['y'] + char['h']), (0, 255, 0), 2)
            debug_windows["2_Segmented_Characters"] = contour_debug_image

        # --- 3. Final OCR on the Assembled Character Line ---
        min_x = min(c['x'] for c in best_line)
        min_y = min(c['y'] for c in best_line)
        max_x = max(c['x'] + c['w'] for c in best_line)
        max_y = max(c['y'] + c['h'] for c in best_line)

        # Crop the entire line of characters from the processed plate
        line_image = dilated_plate[min_y:max_y, min_x:max_x]
        
        # Invert colors for Tesseract and apply a final polish
        inverted_line = cv2.bitwise_not(line_image)
        polished_line = cv2.medianBlur(inverted_line, 3)
        padded_line = cv2.copyMakeBorder(polished_line, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        if debug:
            debug_windows["3_Final_OCR_Strip"] = padded_line

        # "Golden Configuration" for Tesseract on license plates
        custom_config = r'--oem 3 --psm 8 -c load_system_dawg=0 -c load_freq_dawg=0'
        
        ocr_text = pytesseract.image_to_string(padded_line, config=custom_config)
        cleaned_text = "".join(c for c in ocr_text if c.isalnum()).upper()
        
        if cleaned_text:
            detections.append({"text": cleaned_text, "confidence": yolo_confidence, "bbox": (x1, y1, x2, y2)})

    if not detections:
        return image, None

    # Select the detection with the highest confidence from YOLO
    best_detection = max(detections, key=lambda d: d['confidence'])
    
    # Prepare the final annotated image
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

# Main block for direct, interactive testing of this script
if __name__ == "__main__":
    image_path = 'car_image_california.jpg' # Change this to test different images
    
    # Run with full visualization
    final_image, result = recognize_plate(image_path, debug=True, display_windows=True)

    if result:
        print(f"\n--- SUCCESS ---")
        print(f"Recognized Plate Text: {result['text']}")
    else:
        print("\n--- FAILURE ---")