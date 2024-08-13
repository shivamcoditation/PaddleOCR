import pandas as pd
from multiprocessing import Pool, cpu_count

df = pd.read_csv("df_combined.csv")

# Drop the unnecessary 'Unnamed: 0' column
df = df.drop(columns=['Unnamed: 0'])

def get_iou(dict1, dict2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x_left = max(dict1['x1'], dict2['x1'])
    y_top = max(dict1['y1'], dict2['y1'])
    x_right = min(dict1['x2'], dict2['x2'])
    y_bottom = min(dict1['y2'], dict2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (dict1['x2'] - dict1['x1']) * (dict1['y2'] - dict1['y1'])
    bb2_area = (dict2['x2'] - dict2['x1']) * (dict2['y2'] - dict2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou_bbox(dict1, dict2):
    """Get the bounding box of the intersection area."""
    x_left = max(dict1['x1'], dict2['x1'])
    y_top = max(dict1['y1'], dict2['y1'])
    x_right = min(dict1['x2'], dict2['x2'])
    y_bottom = min(dict1['y2'], dict2['y2'])
    return x_left, x_right, y_top, y_bottom

def check_text_inside_bbox(bbox1, bbox2, area_around=2):
    x, y, w, h = bbox1
    x, y, w, h = x - area_around, y - area_around, w + area_around * 2, h + area_around * 2
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    dict1 = {'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h}
    dict2 = {'x1': x2, 'x2': x2 + w2, 'y1': y2, 'y2': y2 + h2}

    if get_iou(dict1, dict2) > 0:
        a, b, c, d = get_iou_bbox(dict1, dict2)
        area_per = (abs(a - b) * abs(c - d)) / (w2 * h2)
        if area_per > 0.5:
            return True
    return False

def check_boxes(i, j):
    if i > j:
        box1 = df.iloc[i]
        box2 = df.iloc[j]
        bbox1 = (box1['x'], box1['y'], box1['w'], box1['h'])
        bbox2 = (box2['x'], box2['y'], box2['w'], box2['h'])
        if check_text_inside_bbox(bbox1, bbox2):
            if box1['w'] * box1['h'] >= box2['w'] * box2['h']:
                return j
            else:
                print(box1['text'],box2['text'])
                return i
    return None

def filter_boxes(df):
    to_keep = set(range(len(df)))
    pairs = [(i, j) for i in range(len(df)) for j in range(len(df))]

    with Pool(cpu_count()) as pool:
        results = pool.starmap(check_boxes, pairs)

    to_discard = set(filter(lambda x: x is not None, results))
    to_keep -= to_discard

    return df.iloc[list(to_keep)].reset_index(drop=True)

if __name__ == '__main__':
    # Your code that loads the dataframe (df) here
    df = pd.read_csv('df_combined.csv')  # Example
    print(len(df))
    # Wrap the multiprocessing part
    with Pool(cpu_count()) as pool:
        df_cleaned = filter_boxes(df)
    print(len(df_cleaned))
    # Output the cleaned DataFrame
    df_cleaned.to_csv("df_cleaned.csv")
