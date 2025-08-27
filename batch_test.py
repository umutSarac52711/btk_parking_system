# batch_test.py

import os
import csv
from main import recognize_plate_two_pass # IMPORTANT: Assumes your main function is in 'main.py'

# --- Configuration ---
# Point this to the folder where your dataset was downloaded
dataset_folder = 'dataset'
# This is the file where we will save our results
output_csv_file = 'results.csv'

# --- The Main Loop ---
print("Starting batch processing...")

# Prepare the CSV file
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    # Create a writer object and write the header row
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image_path', 'detected_text', 'score'])

    # os.walk is a great way to go through all files in a directory and its subdirectories
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            # Check if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                print(f"\n--- Processing: {image_path} ---")

                try:
                    # Run your main recognition function
                    # We modify it to return the result instead of displaying it
                    final_image, best_candidate = recognize_plate_two_pass(image_path, display_windows=False)
                    
                    if best_candidate:
                        detected_text = best_candidate['text']
                        score = best_candidate['score']
                        print(f"  -> SUCCESS: Found '{detected_text}'")
                    else:
                        detected_text = "NONE_FOUND"
                        score = 0
                        print(f"  -> FAILURE: No plate found.")
                    
                    # Write the result to our CSV file
                    csv_writer.writerow([image_path, detected_text, score])

                except Exception as e:
                    print(f"  -> ERROR processing this image: {e}")
                    csv_writer.writerow([image_path, 'ERROR', 0])


print(f"\n--- Batch Processing Complete! ---")
print(f"Results saved to {output_csv_file}")