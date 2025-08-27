import easyocr
import cv2
import numpy as np
import re

# (Helper functions get_plate_patterns, is_valid_plate, calculate_area remain the same)
def get_plate_patterns():
    patterns = [ r"^[A-Z0-9]{5,8}$", r"^\d{2}[A-Z]{1,3}\d{2,4}$" ]
    return patterns

def is_valid_plate(text, patterns):
    has_digit = any(char.isdigit() for char in text)
    has_letter = any(char.isalpha() for char in text)
    if not (has_digit and has_letter) and len(text) > 4:
        return False
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    return False

def calculate_area(bounding_box):
    return cv2.contourArea(np.array(bounding_box, dtype=np.int32))

# --- THE FIX: NEW PREPROCESSING WITH CLAHE ---
def preprocess_for_ocr(image):
    """
    Enhances the image for OCR using CLAHE (Contrast Limited Adaptive Histogram Equalization),
    which improves local contrast while preserving a natural look.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Create a CLAHE object.
    # clipLimit: This is the threshold for contrast limiting. A value of 2.0 is standard.
    # tileGridSize: Divides the image into an 8x8 grid for local processing.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 2. Apply the CLAHE object to our grayscale image.
    enhanced_gray = clahe.apply(gray)
    
    return enhanced_gray


def recognize_plate_two_pass(image_path):
    # This entire function remains the same as the last version.
    # It will now just call our new and improved preprocess_for_ocr.
    image = cv2.imread(image_path)
    if image is None: return

    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)

    print("\n--- Pass 1: Finding all text regions ---")
    pass1_results = reader.readtext(image, paragraph=False)
    print(f"  -> Found {len(pass1_results)} regions.")

    plate_patterns = get_plate_patterns()
    final_candidates = []
    debug_windows = []

    print("\n--- Pass 2: Re-scanning and validating each region ---")
    for i, (bbox, text, prob) in enumerate(pass1_results):
        print(f"\nProcessing Region #{i+1} ('{text}')")
        
        all_points = np.array(bbox, dtype=int)
        min_x, min_y, max_x, max_y = np.min(all_points[:,0]), np.min(all_points[:,1]), np.max(all_points[:,0]), np.max(all_points[:,1])
        padding = 5
        cropped_region = image[max(0, min_y-padding):min(image.shape[0], max_y+padding), max(0, min_x-padding):min(image.shape[1], max_x+padding)]
        if cropped_region.size == 0: continue
            
        preprocessed_crop = preprocess_for_ocr(cropped_region) # This now uses CLAHE
        window_name = f"Region #{i+1} Processed ('{text}')"
        debug_windows.append((window_name, preprocessed_crop))

        pass2_results = reader.readtext(preprocessed_crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ012356789') # Added 4 back
        
        if not pass2_results:
            print("  -> No text found after preprocessing.")
            continue

        for (_, refined_text, refined_prob) in pass2_results:
            cleaned_text = refined_text.upper().replace(" ", "")
            print(f"  -> Re-scanned and found: '{cleaned_text}'")
            
            if is_valid_plate(cleaned_text, plate_patterns):
                area = calculate_area(bbox)
                score = area * refined_prob
                final_candidates.append({ "text": cleaned_text, "score": score, "bbox": bbox })
                print(f"    -> ACCEPTED: Valid plate format. Score: {int(score)}")
            else:
                print(f"    -> REJECTED: Invalid plate format.")

    for name, img in debug_windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)

    if final_candidates:
        best_candidate = max(final_candidates, key=lambda c: c["score"])
        plate_text = best_candidate["text"]
        bounding_box = best_candidate["bbox"]
        print(f"\n--- FINAL RESULT: SUCCESS ---")
        print(f"Selected Best Candidate: '{plate_text}' (Score: {int(best_candidate['score'])})")
        display_image = image.copy()
        all_points = np.array(bounding_box, dtype=int)
        min_x, min_y, max_x, max_y = np.min(all_points[:,0]), np.min(all_points[:,1]), np.max(all_points[:,0]), np.max(all_points[:,1])
        cv2.rectangle(display_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
        cv2.putText(display_image, plate_text, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Result", display_image)
    else:
        print("\n--- FINAL RESULT: FAILURE ---")

    print("\nPress any key to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = 'car_image_california.jpg' 
    recognize_plate_two_pass(image_path)

if __name__ == "__main__":
    main()