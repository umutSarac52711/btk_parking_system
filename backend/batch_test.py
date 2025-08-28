# batch_test.py

import os
import csv
# Make sure to import from your final, working script
from backend.services.recognition_service import recognize_plate

# --- Configuration ---
# Point this to the folder where your dataset was downloaded
dataset_folder = 'dataset' # The new downloader uses 'dataset' by default
output_csv_file = 'yolo_results.csv'

# --- The Main Loop ---
print(f"Starting batch processing of images in '{dataset_folder}'...")

# Prepare the CSV file
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image_path', 'detected_text', 'yolo_confidence'])

    # Loop through all files in the dataset folder and its subdirectories
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                print(f"\n--- Processing: {image_path} ---")

                try:
                    # Call the recognition function with the UI turned OFF
                    final_image, best_detection = recognize_plate(image_path, debug=False, display_windows=False)
                    
                    if best_detection:
                        # If a detection was returned, extract its data
                        detected_text = best_detection['text']
                        # YOLO's confidence is a good score to log
                        score = best_detection['confidence']
                        print(f"  -> SUCCESS: Found '{detected_text}' with confidence {score:.2f}")
                        csv_writer.writerow([image_path, detected_text, f"{score:.2f}"])
                    else:
                        # This handles the case where the function returns None
                        print(f"  -> FAILURE: No plate found.")
                        csv_writer.writerow([image_path, 'NONE_FOUND', 0.0])

                except Exception as e:
                    # This catches any unexpected crashes during processing
                    print(f"  -> ERROR processing this image: {e}")
                    csv_writer.writerow([image_path, 'ERROR', 0.0])

print(f"\n--- Batch Processing Complete! ---")
print(f"Results saved to {output_csv_file}")