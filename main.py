import easyocr
import cv2
import numpy as np
import re

# (Helper functions get_plate_patterns, calculate_area remain the same)
def get_plate_patterns():
    patterns = [ r"^[A-Z0-9]{5,8}$", r"^\d{2}[A-Z]{1,3}\d{2,4}$" ]
    return patterns
def calculate_area(bounding_box):
    return cv2.contourArea(np.array(bounding_box, dtype=np.int32))

# --- THE FIX #1: SMARTER VALIDATION ---
def is_valid_plate(text, patterns):
    """
    Checks if a text string is a valid plate.
    Rejects common words but allows all-letter vanity plates.
    """
    # Rule 1: Reject obvious long words that contain no numbers.
    if len(text) >= 7 and text.isalpha():
        # You could add a dictionary check here for more robustness,
        # but for now, this will reject "CALIFORNIA" but allow "MYLILPY"
        # if we adjust the length check slightly or add another rule.
        # Let's refine: if it's all letters, it can't be too long.
        if len(text) > 7:
             return False # Rejects "CALIFORNIA", allows "MYLILPY"

    # Rule 2: Must match one of our regex patterns.
    for pattern in patterns:
        if re.match(pattern, text):
            return True
            
    return False

# We now have TWO preprocessing functions
def preprocess_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def recognize_plate_two_pass(image_path, display_windows=True):
    image = cv2.imread(image_path)
    if image is None: return

    reader = easyocr.Reader(['en'], gpu=False)
    pass1_results = reader.readtext(image, paragraph=False)
    print(f"\n--- Found {len(pass1_results)} initial regions ---")

    plate_patterns = get_plate_patterns()
    final_candidates = []

    # --- THE FIX #2: ENSEMBLE PREPROCESSING ---
    preprocessing_methods = {
        "CLAHE": preprocess_clahe,
        "Grayscale": preprocess_grayscale
    }

    for i, (bbox, text, prob) in enumerate(pass1_results):
        print(f"\nProcessing Region #{i+1} ('{text}')")
        
        all_points = np.array(bbox, dtype=int)
        min_x, min_y, max_x, max_y = np.min(all_points[:,0]), np.min(all_points[:,1]), np.max(all_points[:,0]), np.max(all_points[:,1])
        padding = 5
        cropped_region = image[max(0, min_y-padding):min(image.shape[0], max_y+padding), max(0, min_x-padding):min(image.shape[1], max_x+padding)]
        if cropped_region.size == 0: continue

        # Now, loop through our different preprocessing methods
        for method_name, preprocess_func in preprocessing_methods.items():
            preprocessed_crop = preprocess_func(cropped_region)
            
            pass2_results = reader.readtext(preprocessed_crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if not pass2_results: continue

            # We only care about the dominant text from each scan
            dominant_text_in_scan = max(pass2_results, key=lambda r: calculate_area(r[0]))
            
            cleaned_text = dominant_text_in_scan[1].upper().replace(" ", "")
            refined_prob = dominant_text_in_scan[2]
            
            print(f"  -> Scanned with [{method_name}]: Found '{cleaned_text}' (Conf: {refined_prob:.2f})")

            # Use our new, smarter validation function
            if is_valid_plate(cleaned_text, plate_patterns):
                area = calculate_area(bbox)
                score = area * refined_prob
                final_candidates.append({ "text": cleaned_text, "score": score, "bbox": bbox })
                print(f"    -> ACCEPTED: Valid plate format. Score: {int(score)}")
            else:
                print(f"    -> REJECTED: Invalid plate format.")

    # (Selection logic remains the same, but now it has more candidates to choose from)
    if not final_candidates:
        print("\n--- FINAL RESULT: FAILURE ---")
        return

    best_candidate = max(final_candidates, key=lambda c: c["score"])
    plate_text, bounding_box = best_candidate["text"], best_candidate["bbox"]
    
    print(f"\n--- FINAL RESULT: SUCCESS ---")
    print(f"Selected Best Candidate: '{plate_text}' (Score: {int(best_candidate['score'])})")
    
    # At the very end, instead of just displaying, check the flag
    if display_windows:
        # (all of your cv2.imshow, cv2.waitKey calls go here)
        # ...
        # (Drawing code is the same)
        display_image = image.copy()
        all_points = np.array(bounding_box, dtype=int)
        min_x, min_y, max_x, max_y = np.min(all_points[:,0]), np.min(all_points[:,1]), np.max(all_points[:,0]), np.max(all_points[:,1])
        cv2.rectangle(display_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
        cv2.putText(display_image, plate_text, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Result", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Crucially, return the result so the batch script can use it
    if final_candidates:
        return display_image, max(final_candidates, key=lambda c: c["score"])
    else:
        return image, None # Return None if nothing was found

    

    


def main():
    # Test both images!
    print("--- TESTING CALIFORNIA PLATE ---")
    recognize_plate_two_pass('car_image_california.jpg')
    print("\n\n--- TESTING WEST VIRGINIA PLATE ---")
    recognize_plate_two_pass('car_image.jpg')


if __name__ == "__main__":
    main()