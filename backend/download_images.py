# download_images.py

# We now import from the new, better library
from bing_image_downloader import downloader
import os

# --- Strategic Search Queries ---
# This list remains the same. The new library will handle them correctly.
search_queries = [
    'car parked on street rear view',
    'car license plate close up',
    'license plate on truck',
    'cars in traffic jam',
    'Türkiye araba plakası',
    'car rear bumper',
    'European license plate on car',
    'US license plate on car'
]

# --- The Downloading Loop ---
number_of_images_per_query = 20

for query in search_queries:
    print(f"Downloading {number_of_images_per_query} images for query: '{query}'")
    try:
        # --- The new, more robust download command ---
        downloader.download(
            query,
            limit=number_of_images_per_query,
            output_dir='dataset',  # All images will go into a single 'dataset' folder
            adult_filter_off=True,
            force_replace=False,
            timeout=60, # Increased timeout for slow connections
            verbose=True # Shows download progress for each file
        )
        print(f"Successfully downloaded images for '{query}'")
    except Exception as e:
        print(f"An error occurred while downloading for '{query}': {e}")
        continue

print("\n--- Download Complete! ---")
# The images are now in 'dataset/<query_name>/'
print("Check the 'dataset' folder in your project directory.")