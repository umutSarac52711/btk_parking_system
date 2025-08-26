import cv2 # Import the OpenCV library for image processing

def main():
    image_path = 'car_image.jpg'
    print(f"Attempting to load image from: {image_path}") # This is an f-string, an easy way to format strings.

    # We use the cv2.imread() function to read an image from the file path.
    # OpenCV loads the image as a NumPy array - essentially a multi-dimensional
    # grid of numbers where each number represents a pixel's color.
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image. Check if the file '{image_path}' exists.")
        return 

    # cv2.imshow() takes two arguments:
    # 1. The name of the window (a string).
    # 2. The image data we want to display.
    cv2.imshow('Car Image', image)

    # cv2.waitKey(0) tells the program to pause and wait indefinitely for a key
    # press. If you don't have this, the window will appear and disappear
    # instantly, and you won't see anything. The '0' means wait forever.
    # If you gave it a number like 1000, it would wait for 1 second (1000 ms).
    print("Image displayed. Press any key to close the window.")
    cv2.waitKey(0)

    # After a key is pressed, we should clean up and close all the windows
    # created by OpenCV.
    cv2.destroyAllWindows()
    print("Window closed. Program finished.")

if __name__ == "__main__":
    main()