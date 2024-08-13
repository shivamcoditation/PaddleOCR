import os
import shutil

# Define the source and destination directories
destination_folder = 'text_boxes'
source_folder = 'fraction_images'

# Ensure the destination folder exists
# os.makedirs(destination_folder, exist_ok=True)

# Iterate over all files in the source folder
for filename in os.listdir(source_folder):
    # Construct full file path
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(destination_folder, filename)
    
    # Move the file
    shutil.move(source_file, destination_file)

print("All images moved successfully.")

# Paths to the text files
source_file_path = 'mappings.txt'
destination_file_path = 'train_list.txt'

# Open the source file in read mode and the destination file in append mode
with open(source_file_path, 'r', encoding='utf-8') as source_file:
    with open(destination_file_path, 'a', encoding='utf-8') as destination_file:
        # Read the content of the source file
        content = source_file.read()
        # Write the content to the destination file
        destination_file.write(content)

print("Content appended successfully.")

#fraction_178228.png
