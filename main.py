import cv2 as cv
import numpy as np
import os
from pathlib import Path
import pandas as pd
from time import time
from tqdm import tqdm


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
    """ resize and time """
    _start = time()
    resized_image = cv.resize(image, (int(image.shape[1]/n), int(image.shape[0]/n)))
    run_time = time() - _start
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
        optimal_score = min_val
    else:
        top_left = max_loc
        optimal_score = max_val
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    top_left = [x*n for x in top_left]
    return top_left, run_time, resize_image_run_time, resize_template_run_time, optimal_score


def add_roi_rect(img, top_left, shape, linewidth = 8):
    """ Add a rectangle for ROI """
    h, w = shape[0:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    if len(img.shape) == 3:
        cv.rectangle(img, top_left, bottom_right, (0, 0, 255), linewidth) #bgr
    else:
        # need the following line to solve this bug
        img = np.array(img)
        cv.rectangle(img, top_left, bottom_right, 255, linewidth)

    return img


if __name__ == '__main__':

    # factors: method, n, template, raw

    # Blister from raw blister
    # home_folder = "C:/Temp/roi_study_result/blister"
    # template_folder = "C:/Temp/roi_study/blister_front/roi_template"
    # image_folder = "C:/Temp/roi_study/blister_front/normal_raw"

    # Emblossing from carton side
    home_folder = "C:/Temp/roi_study_result/carton_ocr"
    template_folder = "C:/Temp/roi_study/carton_ocr/roi_template"
    image_folder = "C:/Temp/roi_study/carton_ocr/normal_raw"

    # OCR from blister raw
    # home_folder = "C:/Temp/roi_study_result/blister_ocr"
    # template_folder = "C:/Temp/roi_study/blister_front/ocr_roi_template"
    # image_folder = "C:/Temp/roi_study/blister_front/normal_raw"

    # OCR from blister ROI
    # home_folder = "C:/Temp/roi_study_result/blister_ocr_from_roi"
    # template_folder = "C:/Temp/roi_study/blister_front/ocr_roi_template"
    # image_folder = "C:/Temp/roi_study/blister_front/roi_template"

    #method_names = ['cv.TM_CCOEFF']

    method_names = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                     'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    resizing_factors = (1, 2, 4, 8, 16, 32, 64, 128)

    # use part of the images
    roi_templates = list_files_recur(template_folder)[0][0:5]
    normal_raw_imgs = list_files_recur(image_folder)[0][0:20]

    L = []

    # loop over methods
    for method_name in method_names:

        os.makedirs(os.path.join(home_folder, method_name), exist_ok=True)

        # loop over resizing factors
        for n in resizing_factors:

            os.makedirs(os.path.join(home_folder, method_name, str(n)), exist_ok=True)

            # loop over multiple ROI templates
            for template_idx, template_path in enumerate(roi_templates):

                os.makedirs(os.path.join(home_folder, method_name, str(n), Path(template_path).stem), exist_ok=True)

                template = cv.imread(template_path)

                print(method_name, n)

                # loop over raw images
                for raw_image_idx, raw_image_path in enumerate(tqdm(normal_raw_imgs)):

                    # color images
                    img = cv.imread(raw_image_path)
                    top_left, match_time, resize_time_1, resize_time_2, optimal_score = find_reference_point(img, template, n, method_name)
                    #image_with_roi = add_roi_rect(img, top_left, template.shape)
                    #save_path = os.path.join(home_folder, method_name, str(n), Path(template_path).stem, Path(raw_image_path).stem + ".png")
                    #cv.imwrite(save_path, image_with_roi)

                    # grayscale images
                    _start = time()
                    img_gray = img[:, :, 0]
                    template_gray = template[:, :, 0]
                    gray_time = time() - _start
                    top_left_gray, match_time_gray, resize_time_1_gray, resize_time_2_gray, optimal_score_gray = find_reference_point(img_gray, template_gray, n, method_name)
                    #image_with_roi_gray = add_roi_rect(img_gray, top_left_gray, template_gray.shape)
                    #save_path_gray = os.path.join(home_folder, method_name, str(n), Path(template_path).stem, Path(raw_image_path).stem + "_gray.png")
                    #cv.imwrite(save_path_gray, image_with_roi_gray)

                    # save information
                    row = {
                        'method': method_name,
                        'n': n,
                        'template': Path(template_path).stem,
                        'image': Path(raw_image_path).stem,

                        'color_resize_time_image': resize_time_1,
                        'color_resize_time_template': resize_time_2,
                        'color_match_time': match_time,
                        'color_total_time': match_time + resize_time_1 + resize_time_2,
                        'color_x': top_left[0],
                        'color_y': top_left[1],
                        'optimal_score': optimal_score,

                        'gray': gray_time,
                        'gray_resize_time_image': resize_time_1_gray,
                        'gray_resize_time_template': resize_time_2_gray,
                        'gray_match_time': match_time_gray,
                        'gray_total_time': match_time_gray + resize_time_1_gray + resize_time_2_gray,
                        'gray_x': top_left_gray[0],
                        'gray_y': top_left_gray[1],
                        'optimal_score_gray': optimal_score_gray
                    }
                    L.append(row)

    df = pd.DataFrame(L)
    df.to_csv(os.path.join(home_folder, "summary.csv"), index=False)
