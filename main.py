import easyocr

# This only needs to be done once
reader = easyocr.Reader(['en']) # 'en' for English, 'tr' for Turkish, etc.

# Run this on your image
results = reader.readtext('car_image.jpg')

# Process the results
for (bbox, text, prob) in results:
    # bbox is the coordinates of the license plate
    # text is the recognized license plate string
    print(f'Found text: "{text}" with probability {prob}')