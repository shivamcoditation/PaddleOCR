import cv2
import pandas as pd
import os
import numpy as np
import json
import concurrent.futures
from PIL import Image
import uuid

def convert(df, filename):
    # Create a list to store annotations for this image
    annotations = []

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        # Extracting relevant information from the DataFrame
        transcription = row['text']
        x, y, w, h = row['x'], row['y'], row['w'], row['h']
        # Assuming the points are represented as a list of lists [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        points = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]

        # Create a dictionary for this annotation
        annotation = {"transcription": transcription, "points": points}

        # Append this annotation to the list of annotations
        annotations.append(annotation)

    # Create a dictionary to hold the image annotations
    image_annotations = {
        "image_path": f"{filename}",  # Adjust the path according to your image location
        "annotations": annotations
    }
    return image_annotations

def rotate_coordinates(entry, image_width, image_height):
    new_x = image_height - entry['y'] - entry['h']
    new_y = entry['x']
    new_w = entry['h']
    new_h = entry['w']
    return new_x, new_y, new_w, new_h

def process_image_detection(folder_path):
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                json_path = os.path.join(folder_path, filename)
                # Corresponding image filename
                image_filename = filename.replace('.json', '.png')
                image_path = os.path.join(folder_path, image_filename)
                image = cv2.imread(image_path)
                image_copy = image.copy()
                patch_size = (1000, 1000)  # Set the size of each patch (width, height)
                stride = (900, 900)  # Set the stride (horizontal, vertical)
                num_patches_height = int((image.shape[0] - patch_size[1]) / stride[1]) + 1
                num_patches_width = int((image.shape[1] - patch_size[0]) / stride[0]) + 1
                with open(json_path, 'r', encoding='utf-8') as f:
                    data_ = json.load(f)
                textdf = pd.DataFrame(data_)
                textdf = textdf[~textdf.text.isna()]
                textdf = textdf[textdf['text'].str.strip() != '']
                image_filename = image_filename.replace(".pdf","pdf")
                image_filename = image_filename.replace("(","")
                image_filename = image_filename.replace(")","")
                image_filename = image_filename.replace("-","_")
                image_filename = image_filename.replace(" ","")
                image_filename = image_filename.replace("\xef\xbb\xbf","")
                image_filename = image_filename.rstrip()
                image_filename = image_filename.lstrip()
                image_filename = image_filename.replace(".png","")
                image_filename = image_filename.replace(".jpg","")
                image_filename = image_filename.replace(".jpeg","")
                # Iterate through the patches
                for i in range(num_patches_height):
                    for j in range(num_patches_width):
                        
                        y_start = i * stride[1]
                        y_end = y_start + patch_size[1]
                        x_start = j * stride[0]
                        x_end = x_start + patch_size[0]
                        patch = image[y_start:y_end, x_start:x_end]
                    
                        if not np.all(patch == 255):
                            # horizontal
                            patch_filename = f"{image_filename}_patch_{i}_{j}_horizontal.jpg"
                            patch_filepath = os.path.join("patches_detector", patch_filename)
                            temp_df_h = pd.DataFrame(columns=['text', 'x', 'y', 'w', 'h'])
                            for index, row in textdf.iterrows():
                                if row['orientation'] == "horizontal":
                                    x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
                                    if x>=x_start and y>=y_start and (x+w)<=x_end and (y+h)<=y_end:
                                        x = x - x_start
                                        y = y - y_start                   
                                        row_df = pd.DataFrame([[row['text'], x, y, w, h]], columns=['text', 'x', 'y', 'w', 'h'])
                                        temp_df_h = pd.concat([temp_df_h, row_df], ignore_index=True)
                                        # cv2.rectangle(patch, (x, y), ((x + w), (y + h)), (0, 0, 255), 5)
                            cv2.imwrite(patch_filepath,patch)
                            data = convert(temp_df_h,patch_filename)
                            with open('mappings_detection.txt', 'a') as file:
                                file.write(data['image_path'] + "\t" + json.dumps(data['annotations'])+ '\n')
                                
                            # vertical
                            patch = image_copy[y_start:y_end, x_start:x_end]
                            patch_filename = f"{image_filename}_patch_{i}_{j}_vertical.jpg"
                            patch_filepath = os.path.join("patches_detector", patch_filename)
                            rotated_patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
                            temp_df_v = pd.DataFrame(columns=['text', 'x', 'y', 'w', 'h'])
                            for index, row in textdf.iterrows():
                                if row['orientation'] == "vertical":
                                    x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
                                    if x>=x_start and y>=y_start and (x+w)<=x_end and (y+h)<=y_end:
                                        x = x - x_start
                                        y = y - y_start
                                        new_x, new_y, new_w, new_h = rotate_coordinates({"x": x, "y": y, "w": w, "h": h}, patch.shape[1], patch.shape[0])
                                        row_df = pd.DataFrame([[row['text'],new_x,new_y,new_w,new_h]],columns=['text', 'x', 'y', 'w', 'h'])
                                        temp_df_v = pd.concat([temp_df_v, row_df], ignore_index=True)
                                        # cv2.rectangle(rotated_patch, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 255), 2)
                            cv2.imwrite(patch_filepath,rotated_patch)
                            data = convert(temp_df_v,patch_filename)
                            with open('mappings_detection.txt', 'a') as file:
                                file.write(data['image_path'] + "\t" + json.dumps(data['annotations'])+ '\n') 
            print(f"Completed for {filename}")

    except Exception as e:
        print(e) 

def process_image_recognition(folder_path):
    try:
        # Create a text file to store the mappings
        with open('mappings_recognizer.txt', 'w', encoding='utf-8') as mappings_file:
            for json_filename in os.listdir(folder_path):
                if json_filename.endswith('.json'):
                    json_path = os.path.join(folder_path, json_filename)

                    # Corresponding image filename
                    image_filename = json_filename.replace('.json', '.png')
                    image_path = os.path.join(folder_path, image_filename)

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
                            cropped_image = image.crop((x-2, y-2, x + w+4, y + h+4))

                            # Save the cropped image with the UUID as the filename
                            cropped_image_filename = f'{uuid_str}.png'
                            cropped_image.save("patches_recognizor/" + cropped_image_filename)

                            # Write the mapping to the text file
                            mappings_file.write(f'{cropped_image_filename}\t{text}\n')
                        print("Done")    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    folder_path = "All_images"

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Schedule both image detection and recognition tasks
        detection_futures = [executor.submit(process_image_detection, folder_path)]
        recognition_futures = [executor.submit(process_image_recognition, folder_path)]

        # Wait for all tasks to complete
        concurrent.futures.wait(detection_futures + recognition_futures)

    print("All image processing tasks completed.")