import easyocr
import cv2
import numpy as np

def recognize_plate_easyocr(image_path):
    """
    This function uses easyocr to find and read a license plate from an image.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at '{image_path}'")
        return None, None

    # --- EASYOCR MAGIC HAPPENS HERE ---

    # 1. Initialize the reader.
    # We specify 'tr' for Turkish and 'en' for English. It will use both.
    # The 'gpu=False' flag is important if you don't have a CUDA-enabled GPU.
    # The model is downloaded automatically the first time you run this.
    print("Initializing EasyOCR reader... (This may take a moment on first run)")
    reader = easyocr.Reader(['tr', 'en'], gpu=False)

    # 2. Perform OCR on the image.
    # The 'detail=1' is the default and gives bounding boxes.
    # 'paragraph=False' tells it to treat distinct blocks of text as separate.
    results = reader.readtext(image, paragraph=False)
    print(f"EasyOCR found {len(results)} text blocks.")

    # --- PROCESS THE RESULTS ---
    
    # We will assume the most likely candidate is the one we want.
    # You could add more logic here later (e.g., check if text matches plate format).
    if not results:
        print("No text found.")
        return image, None

    # Let's find the result with the highest confidence score.
    best_result = max(results, key=lambda r: r[2]) # r[2] is the confidence score
    
    plate_text = best_result[1]
    confidence = best_result[2]
    bounding_box = best_result[0]

    print(f"Best guess for plate: '{plate_text}' with confidence {confidence:.2f}")

    # Draw the bounding box and text on the image.
    display_image = image.copy()
    
    # The bounding_box is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    # We need to find the top-left and bottom-right points to draw a simple rectangle.
    # (Note: easyocr provides all 4 corners, so it can handle rotated text!)
    top_left = tuple(map(int, bounding_box[0]))
    bottom_right = tuple(map(int, bounding_box[2]))
    
    # For drawing, it's safer to get the min/max of all points in case of rotation
    all_points = np.array(bounding_box, dtype=int)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    # Draw the rectangle
    cv2.rectangle(display_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)

    # Put the recognized text above the rectangle
    cv2.putText(display_image, plate_text, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return display_image, plate_text

def main():
    image_path = 'car_image.jpg' # Use a clear image for the best result
    
    final_image, plate_text_result = recognize_plate_easyocr(image_path)

    if final_image is not None:
        if plate_text_result:
            print(f"\n--- SUCCESS ---")
            print(f"Recognized Plate Text: {plate_text_result}")
        else:
            print("\n--- FAILURE ---")
            print("Could not recognize a license plate.")

        # Display the final image
        cv2.namedWindow("EasyOCR Result", cv2.WINDOW_NORMAL)
        cv2.imshow("EasyOCR Result", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()