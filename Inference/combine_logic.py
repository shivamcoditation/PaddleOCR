import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from clean import filter_boxes
def are_close(box1, box2):
    if box1['orientation'] != box2['orientation']:
        return False
    if box1['orientation'] == 'horizontal':
        # Check if boxes are close horizontally and vertically
        horizontal_close = abs((box1['x'] + box1['w']) - box2['x']) <= 30 or abs((box2['x'] + box2['w']) - box1['x']) <= 30
        vertical_close = abs(box1['y'] - box2['y']) <= 5
        return horizontal_close and vertical_close
    else:
        vertical_close = abs((box1['y'] + box1['h']) - box2['y']) <= 30 or abs((box2['y'] + box2['h']) - box1['y']) <= 30
        horizontal_close = abs(box1['x'] - box2['x']) <= 5
        return vertical_close and horizontal_close

def merge_once(df_chunk):
    combined_boxes = []
    skip_indices = set()

    for i, box1 in df_chunk.iterrows():
        if i in skip_indices:
            continue
        combined_box = box1.copy()
        for j, box2 in df_chunk.iterrows():
            if i != j and j not in skip_indices and are_close(box1, box2):
                combined_box['text'] += " " + box2['text']
                combined_box['x'] = min(combined_box['x'], box2['x'])
                combined_box['y'] = min(combined_box['y'], box2['y'])
                combined_box['w'] = max(box1['x'] + box1['w'], box2['x'] + box2['w']) - combined_box['x']
                combined_box['h'] = max(box1['y'] + box1['h'], box2['y'] + box2['h']) - combined_box['y']
                skip_indices.add(j)
        combined_boxes.append(combined_box)

    return pd.DataFrame(combined_boxes).reset_index(drop=True)

def combine_boxes(df, num_threads=4):
    prev_len = -1
    current_len = len(df)

    while current_len != prev_len:
        # Split dataframe into chunks for parallel processing
        df_chunks = np.array_split(df, num_threads)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(lambda chunk: merge_once(chunk), df_chunks))

        # Concatenate results and drop duplicates
        df = pd.concat(results).drop_duplicates().reset_index(drop=True)
        prev_len = current_len
        current_len = len(df)
        print(len(df))
    return df

if __name__ == '__main__':
    
    df = pd.read_csv("df_cleaned.csv")
    img = cv2.imread("10687001.jpg")
    df = df.drop(columns=['Unnamed: 0'])
    # Combine boxes
    df_combined = combine_boxes(df)
    # df_cleaned = filter_boxes(df_combined)
    print(len(df_combined))
    for index, row in df_combined.iterrows():
        text = row['text']
        x = int(row['x'])
        y = int(row['y'])
        w = int(row['w'])
        h = int(row['h'])
        orientation = row['orientation']

        # Draw the rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
        # Add the text
        text_pos = (x, y - 5)
        cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Save the image
    cv2.imwrite('combine_clean.png', img)

