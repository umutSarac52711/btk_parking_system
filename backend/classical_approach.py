import cv2
import numpy as np

def find_license_plate(image):
    """
    This function takes an image and finds the license plate,
    returning all intermediate images for debugging.
    """
    debug_images = {}
    img_copy = image.copy()
    debug_images['original'] = img_copy

    # 1. PREPROCESSING
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 100)
    debug_images['1_edged'] = edged # Renamed for ordering

    # --- NEW STEP: MORPHOLOGICAL CLOSING ---
    # This is the key to fixing discontinuous edges.
    # First, we define a "kernel", which is the shape and size of the
    # structuring element used for dilation/erosion. A 5x5 rectangle is a good start.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Now we perform the closing operation.
    # cv2.morphologyEx performs advanced morphological transformations.
    # We give it the edged image, tell it to perform a MORPH_CLOSE, and provide the kernel.
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    debug_images['2_closed'] = closed # Add this to our debug views!

    # 3. CONTOUR FINDING (Now using the 'closed' image)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    debug_images['3_all_contours'] = contour_image

    license_plate_contour = None

    # 4. FILTERING FOR THE PLATE (Now with Aspect Ratio Check)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        # Check if the approximated contour has 4 corners.
        if len(approx) == 4:
            # --- NEW: ASPECT RATIO FILTER ---
            # Get the bounding box of the contour to find its width and height.
            (x, y, w, h) = cv2.boundingRect(approx)

            # Calculate the aspect ratio.
            aspect_ratio = w / float(h)

            # We check if the aspect ratio is within a reasonable range.
            # This helps filter out rectangles that are not shaped like license plates.
            # You may need to tune these values (2.5 to 5.5 is a decent guess).
            if aspect_ratio > 2.5 and aspect_ratio < 5.5:
                license_plate_contour = approx
                # We found a good candidate, let's mark it in green on the contour image for debugging.
                cv2.drawContours(debug_images['3_all_contours'], [license_plate_contour], -1, (0, 255, 0), 2)
                break # Exit the loop

    results = {
        "license_plate_contour": license_plate_contour,
        "debug_images": debug_images
    }
    return results

def main():
    # Main function remains the same as the previous debug version.
    image_path = 'car_image.jpg' # Change this to test different images!
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at '{image_path}'")
        return

    detection_results = find_license_plate(image)
    plate_contour = detection_results["license_plate_contour"]
    debug_views = detection_results["debug_images"]

    final_image = image.copy()
    if plate_contour is not None:
        cv2.drawContours(final_image, [plate_contour], -1, (0, 255, 0), 3)
        print("License plate DETECTED.")
    else:
        print("License plate NOT DETECTED.")

    # Display all windows
    for name, img in sorted(debug_views.items()): # sorted() helps keep them in order
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)

    cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Final Result", final_image)

    print("\nDebug windows displayed. Check the '2_closed' window to see the effect.")
    print("Press any key to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Windows closed.")

if __name__ == "__main__":
    main()