import cv2
import numpy as np

def find_license_plate(image):
    """
    This function takes an image and tries to find the license plate contour.
    """
    
    # --- 1. PREPROCESSING ---

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter. This is an advanced blur that is very good at
    # reducing noise while keeping edges sharp.
    # The parameters are: diameter of the pixel neighborhood, color sigma, and space sigma.
    # You don't need to master these parameters, just know they help in noise reduction.
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # --- 2. EDGE DETECTION ---

    # Use the Canny edge detector to find the edges in the image.
    # The two numbers are the lower and upper thresholds. Any gradient between
    # these two values is considered an edge. A common starting ratio is 1:2 or 1:3.
    edged = cv2.Canny(filtered, 30, 200)

    # --- 3. CONTOUR FINDING ---

    # Find all the contours in the edged image.
    # cv2.RETR_TREE retrieves all contours and reconstructs a full hierarchy of nested contours.
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order and keep the top 10.
    # This is a performance optimization. We assume the license plate will be
    # one of the larger contours in the image.
    # 'key=cv2.contourArea' is a Python feature where we provide a function to be used for sorting.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate_contour = None # Initialize to None

    # --- 4. FILTERING FOR THE PLATE ---

    # Loop over our sorted contours.
    for contour in contours:
        # Approximate the contour shape to a simpler polygon.
        # The second parameter (epsilon) is the maximum distance from the original
        # contour to the approximated contour. It's a measure of "accuracy".
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

        # Check if the approximated contour has 4 corners, which is a characteristic of a rectangle.
        if len(approx) == 4:
            # If it has 4 corners, we assume we have found our license plate.
            license_plate_contour = approx
            break # Exit the loop once we've found it.

    return license_plate_contour


def main():
    """
    This is the main function of our script.
    """
    image_path = 'car_image.jpg'
    print(f"Attempting to load image from: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image. Check if the file '{image_path}' exists.")
        return

    # Call our new function to find the license plate.
    plate_contour = find_license_plate(image)

    # We need to handle the case where no plate was found.
    if plate_contour is not None:
        # If a contour was found, draw it on the original image.
        # The arguments are: image to draw on, the contour(s) to draw (as a list),
        # contour index (-1 means draw all), color (in BGR format - Blue, Green, Red), and thickness.
        cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)
        print("License plate detected!")
    else:
        print("Could not find a license plate contour.")

    # Display the final image with the contour drawn on it.
    cv2.imshow('Resulting Image', image)

    print("Image displayed. Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Window closed. Program finished.")


if __name__ == "__main__":
    main()