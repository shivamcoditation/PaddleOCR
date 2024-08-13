
from paddleocr import PaddleOCR
import cv2
import numpy as np
import pandas as pd
from multiprocessing import get_context, Pool
import ast
import time


def process_patch_horizontal(args):
    patch, x_start, y_start, det_model_dir,rec_model_dir,rec_char_dict_path = args
    ocr = PaddleOCR(rec_char_dict_path = rec_char_dict_path,rec_model_dir=rec_model_dir,det_model_dir=det_model_dir, det_db_thresh=0.1, det_db_box_thresh=0.1,show_log=False,use_angle_cls=False,det_db_score_mode = "slow",use_dilation = True)
    patch_results = ocr.ocr(patch, cls=False)
    data = []
    try:
        for k in range(len(patch_results[0])):
            points = np.array(patch_results[0][k][0], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            points[..., 0] += x_start
            points[..., 1] += y_start
            text = patch_results[0][k][1][0]
            x = points[0][0][0]
            y = points[0][0][1]
            w = abs(points[1][0][0] - points[0][0][0])
            h = abs(points[3][0][1] - points[0][0][1])
            orientation = 'horizontal'
            data.append([k, text, x, y, w, h, orientation])
    
    except Exception as e:
        pass
    return data

def process_patch_vertical(args):
    patch, x_start, y_start, det_model_dir,rec_model_dir,rec_char_dict_path,image = args
    ocr = PaddleOCR(rec_char_dict_path = rec_char_dict_path,rec_model_dir=rec_model_dir,det_model_dir=det_model_dir, det_db_thresh=0.1, det_db_box_thresh=0.1,show_log=False,use_angle_cls=False,det_db_score_mode = "slow",use_dilation = True)
    patch_results = ocr.ocr(patch, cls=False)
    data = []
    try:
        for k in range(len(patch_results[0])):
            points = np.array(patch_results[0][k][0], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            points[..., 0] += x_start
            points[..., 1] += y_start

            point1_old = points[0][0][1]
            point2_old = points[1][0][1]
            point3_old = points[2][0][1]
            point4_old = points[3][0][1]
            points[0][0][1],points[0][0][0] = image.shape[0] - points[0][0][0], point1_old
            points[1][0][1],points[1][0][0] = image.shape[0] - points[1][0][0], point2_old
            points[2][0][1],points[2][0][0] = image.shape[0] - points[2][0][0], point3_old
            points[3][0][1],points[3][0][0] = image.shape[0] - points[3][0][0], point4_old

            text = patch_results[0][k][1][0]

            x = points[1][0][0]
            y = points[1][0][1]
            w = abs(points[2][0][0] - points[1][0][0])
            h = abs(points[0][0][1] - points[1][0][1])
            orientation = 'vertical'
            data.append([k, text, x, y, w, h, orientation])
    
    except Exception as e:
        pass
    return data

if __name__ == '__main__':
    start_time = time.time()
    image_path = '10687001.jpg'
    rec_model_dir = "recognition_model"
    det_model_dir = "detection_model"
    rec_char_dict_path = "en_dict.txt"
    image = cv2.imread(image_path)
    patch_size = (1000, 1000)  # Set the size of each patch (width, height)
    stride = (900, 900)  # Set the stride (horizontal, vertical)

    num_patches_height = int((image.shape[0] - patch_size[1]) / stride[1]) + 2
    num_patches_width = int((image.shape[1] - patch_size[0]) / stride[0]) + 2
    data = []
    num_processes = 4
    with get_context("spawn").Pool(processes=num_processes) as pool:
        args_list = []
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                y_start = i * stride[1]
                y_end = y_start + patch_size[1]
                x_start = j * stride[0]
                x_end = x_start + patch_size[0]
                patch = image[y_start:y_end, x_start:x_end]
                args_list.append((patch, x_start, y_start, det_model_dir,rec_model_dir,rec_char_dict_path))
        
        results = pool.map(process_patch_horizontal, args_list)
        for patch_data in results:
            data.extend(patch_data)

    df_h = pd.DataFrame(data, columns=['id', 'text', 'x', 'y', 'w', 'h', 'orientation'])
    print("Total Horizontal Text Detected : ", len(df_h))
    print("Time : ", time.time() - start_time)
    df_h.to_csv("output_df_h.csv")

    for index, row in df_h.iterrows():
        text = row['text']
        x = int(row['x'])
        y = int(row['y'])
        w = int(row['w'])
        h = int(row['h'])
        orientation = row['orientation']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        text_pos = (x, y - 5)
        cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite("horizontal_text_image.png",image)

    image_v = cv2.imread(image_path)
    image_v = cv2.rotate(image_v, cv2.ROTATE_90_CLOCKWISE)
    num_patches_height = int((image_v.shape[0] - patch_size[1]) / stride[1]) + 2
    num_patches_width = int((image_v.shape[1] - patch_size[0]) / stride[0]) + 2
    data = []
    with get_context("spawn").Pool(processes=num_processes) as pool:
        args_list = []
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                y_start = i * stride[1]
                y_end = y_start + patch_size[1]
                x_start = j * stride[0]
                x_end = x_start + patch_size[0]
                patch = image_v[y_start:y_end, x_start:x_end]
                args_list.append((patch, x_start, y_start, det_model_dir,rec_model_dir,rec_char_dict_path,image))
        
        results = pool.map(process_patch_vertical, args_list)
        for patch_data in results:
            data.extend(patch_data)

    df_v = pd.DataFrame(data, columns=['id', 'text', 'x', 'y', 'w', 'h', 'orientation'])
    print("Total Vertical Text Detected : ", len(df_v))
    print("Time : ", time.time() - start_time)
    df_v.to_csv("output_df_v.csv")

    for index, row in df_v.iterrows():
        text = row['text']
        x = int(row['x'])
        y = int(row['y'])
        w = int(row['w'])
        h = int(row['h'])
        orientation = row['orientation']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        text_pos = (x, y - 5)
        cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite("vertical_text_image.png",image)

    df_combined = pd.concat([df_h, df_v], ignore_index=True)
    df_combined.to_csv("df_combined.csv")

