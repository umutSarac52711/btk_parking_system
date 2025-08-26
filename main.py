import cv2
import numpy as np

def find_license_plate(image):
    """
    This function takes an image and finds the license plate,
    returning all intermediate images for debugging.
    """
    # Create a dictionary to store our intermediate images.
    debug_images = {}

    # It's good practice to work on a copy.
    img_copy = image.copy()
    debug_images['original'] = img_copy

    # 1. PREPROCESSING
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    debug_images['grayscale'] = gray

    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    debug_images['filtered'] = filtered

    # 2. EDGE DETECTION
    edged = cv2.Canny(filtered, 30, 200)
    debug_images['edged'] = edged

    # 3. CONTOUR FINDING
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw contours on for visualization.
    # The '3' means we want a 3-channel (color) image, so we can draw in color.
    contour_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Sort contours and keep the top 10.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Draw all of the top 10 contours on our blank image in blue.
    # This will show us what the algorithm is "considering".
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    debug_images['all_contours'] = contour_image

    license_plate_contour = None

    # 4. FILTERING FOR THE PLATE
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4:
            license_plate_contour = approx
            break

    # The final dictionary to be returned.
    results = {
        "license_plate_contour": license_plate_contour,
        "debug_images": debug_images
    }

    return results

def main():
    image_path = 'car_image.jpg' # Change this to test different images!
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image at '{image_path}'")
        return

    # Call the function and get the results.
    detection_results = find_license_plate(image)
    plate_contour = detection_results["license_plate_contour"]
    debug_views = detection_results["debug_images"]

    # Create the final display image.
    final_image = image.copy()

    if plate_contour is not None:
        cv2.drawContours(final_image, [plate_contour], -1, (0, 255, 0), 3)
        print("License plate DETECTED.")
    else:
        print("License plate NOT DETECTED.")

    # --- DISPLAYING ALL THE WINDOWS ---

    # Display each intermediate step in a separate, resizable window.
    for name, img in debug_views.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)

    # Also display the final result.
    cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Final Result", final_image)

    print("\nAll debug windows are displayed.")
    print("Arrange them on your screen to see the pipeline.")
    print("Press any key to close all windows.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Windows closed.")

if __name__ == "__main__":
    main()