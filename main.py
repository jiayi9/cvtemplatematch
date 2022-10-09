import cv2 as cv
import numpy as np
import os
from pathlib import Path
import pandas as pd
from time import time


def list_files_recur(path: str, fmt: str = ".bmp"):
    """List files recursively in a folder with the specified extension name"""
    file_paths = []
    file_names = []
    path = str(path)
    for r, d, f in os.walk(path):
        for file_name in f:
            if fmt in file_name or fmt.lower() in file_name:
                file_paths.append(os.path.join(r, file_name))
                file_names.append(file_name)
    return [file_paths, file_names]


def resize_image(image: np.ndarray, n: int):
    _start = time()
    resized_image = cv.resize(image, (int(image.shape[1]/n), int(image.shape[0]/n)))
    run_time = time() - _start
    print(run_time)
    return resized_image, run_time


def find_reference_point(img: np.ndarray, template: np.ndarray, n:int = 1, method_name: str = 'cv.TM_CCOEFF'):
    """ Find the top left point as the reference point """
    if n != 1:
        img, resize_image_run_time = resize_image(img, n)
        template, resize_template_run_time = resize_image(template, n)
    else:
        resize_image_run_time, resize_template_run_time = 0, 0

    method = eval(method_name)
    # Apply template Matching

    _start = time()
    res = cv.matchTemplate(img, template, method)
    run_time = time() - _start
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    top_left = [x*n for x in top_left]
    return top_left, run_time, resize_image_run_time, resize_template_run_time


def add_roi_rect(img, top_left, shape):
    """ Add a rectangle for ROI """
    h, w = shape[0:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    if len(img.shape) == 3:
        cv.rectangle(img, top_left, bottom_right, (0, 0, 255), 8) #bgr
    else:
        # need the following line to solve this bug
        img = np.array(img)
        cv.rectangle(img, top_left, bottom_right, 255, 8)

    return img


# method -> n -> template -> raw

home_folder = "C:/Temp/roi_study_result"

method_names = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

resizing_factors = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

roi_templates = list_files_recur("C:/Temp/roi_study/blister_front/roi_template")[0]
normal_raw_imgs = list_files_recur("C:/Temp/roi_study/blister_front/normal_raw")[0]


#resizing_factors = (2,1)
#method_names = ['cv.TM_CCOEFF']

# method, n, template, raw_image, ->
# resizing_time_1, resizing_time_2, color_find_time, color_x, color_y, gray_find_time, gray_x, gray_y

L = []

for method_name in method_names:

    os.makedirs(os.path.join(home_folder, method_name), exist_ok=True)

    for n in resizing_factors:

        os.makedirs(os.path.join(home_folder, method_name, str(n)), exist_ok=True)

        for template_idx, template_path in enumerate(roi_templates[0:1]):

            os.makedirs(os.path.join(home_folder, method_name, str(n), Path(template_path).stem), exist_ok=True)

            template = cv.imread(template_path)

            for raw_image_idx, raw_image_path in enumerate(normal_raw_imgs[0:5]):

                # color
                img = cv.imread(raw_image_path)

                top_left, match_time, resize_time_1, resize_time_2 = find_reference_point(img, template, n, method_name)
                image_with_roi = add_roi_rect(img, top_left, template.shape)
                save_path = os.path.join(home_folder, method_name, str(n), Path(template_path).stem, Path(raw_image_path).stem + ".png")
                cv.imwrite(save_path, image_with_roi)

                # gray scale
                _start = time()
                img_gray = img[:, :, 0]
                template_gray = template[:, :, 0]
                gray_time = time() - _start

                top_left_gray, match_time_gray, resize_time_1_gray, resize_time_2_gray = find_reference_point(img_gray, template_gray, n, method_name)
                image_with_roi_gray = add_roi_rect(img_gray, top_left_gray, template_gray.shape)
                save_path_gray = os.path.join(home_folder, method_name, str(n), Path(template_path).stem, Path(raw_image_path).stem + "_gray.png")
                cv.imwrite(save_path_gray, image_with_roi_gray)

                # save information
                L.append({
                    'method': method_name,
                    'n': n,

                    'match_time': match_time,
                    'resize_time_1': resize_time_1,
                    'resize_time_2': resize_time_2,
                    'x': top_left[0],
                    'y': top_left[1],
                    'total_time_color': match_time + resize_time_1 + resize_time_2,

                    'gray': gray_time,

                    'match_time_gray': match_time_gray,
                    'resize_time_1_gray': resize_time_1_gray,
                    'resize_time_2_gray': resize_time_2_gray,
                    'x_gray': top_left_gray[0],
                    'y_gray': top_left_gray[1],

                    'total_time_gray': match_time_gray + resize_time_1_gray + resize_time_2_gray
                })



                #
                # print(method_name, n,
                #       match_time, resize_time_1, resize_time_2, top_left[0], top_left[1] ,"    |   ",  gray_time,
                #       match_time_gray, resize_time_1_gray, resize_time_2_gray, top_left_gray[0], top_left_gray[1])

df = pd.DataFrame(L)
df.to_csv(os.path.join(home_folder, "summary.csv"), index=False)