import json
import uuid
from PIL import Image
import os

# Ensure the output directory exists
os.makedirs('text_boxes', exist_ok=True)

# Directories containing JSON files and images
json_dir = 'data'
image_dir = 'data'

# Create a text file to store the mappings
with open('mappings_data.txt', 'w', encoding='utf-8') as mappings_file:
    # Iterate over all JSON files in the json_dir
    for json_filename in os.listdir(json_dir):
        if json_filename.endswith('.json'):
            json_path = os.path.join(json_dir, json_filename)

            # Corresponding image filename
            image_filename = json_filename.replace('.json', '.png')
            image_path = os.path.join(image_dir, image_filename)

            # Check if the corresponding image file exists
            if os.path.exists(image_path):
                # Load the JSON data
                with open(json_path, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)

                # Load the image
                image = Image.open(image_path)

                # Process each item in the JSON data
                for item in ocr_data:
                    # Extract the text and coordinates
                    text = item['text']
                    x, y, w, h = item['x'], item['y'], item['w'], item['h']

                    # Generate a unique UUID for this text box
                    uuid_str = str(uuid.uuid4())

                    # Crop the image to the bounding box of the text
                    cropped_image = image.crop((x, y, x + w, y + h))

                    # Save the cropped image with the UUID as the filename
                    cropped_image_filename = f'{uuid_str}.png'
                    cropped_image.save("text_boxes/" + cropped_image_filename)

                    # Write the mapping to the text file
                    mappings_file.write(f'{cropped_image_filename}\t{text}\n')
                print("Done")    
print("Processing completed.")