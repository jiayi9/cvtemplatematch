from main import list_files_recur, resize_image, find_reference_point, add_roi_rect
import cv2 as cv
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

#Blister from raw blister
home_folder = "C:/Temp/roi_study_result/rotate_study"
template_folder = "C:/Temp/roi_study/blister_front/roi_template"
good_image_folder = "C:/Temp/roi_study/blister_front/normal_raw"

good_normal_images = list_files_recur(good_image_folder)[0]

angles = [5, 10, 15, 20, 30, 45, 60, 90]

###################################################################################################################


method_names = ['cv.TM_CCOEFF_NORMED']
resizing_factors = [1]
roi_templates = list_files_recur(template_folder)[0][0:1]

L = []

# loop over methods
for method_name in method_names:
    os.makedirs(os.path.join(home_folder, method_name), exist_ok=True)

    # loop over resizing factors
    for n in resizing_factors:
        os.makedirs(os.path.join(home_folder, method_name, str(n)), exist_ok=True)
        print(n)

        # loop over multiple ROI templates
        for template_idx, template_path in enumerate(roi_templates):
            os.makedirs(os.path.join(home_folder, method_name, str(n), Path(template_path).stem), exist_ok=True)

            template = cv.imread(template_path)

            # normal images
            for path in good_normal_images[0:1]:
                img = cv.imread(path)
                top_left, match_time, resize_time_1, resize_time_2, optimal_score = \
                    find_reference_point(img, template, n, method_name)
                image_with_roi = add_roi_rect(img, top_left, template.shape, linewidth=12)
                save_path = os.path.join(home_folder, f"0_" + ".png")
                cv.imwrite(save_path, image_with_roi)

            # rotated image folders
            for angle in angles:
                print(angle)
                angle_group = "C:/Temp/roi_study/blister_front/normal_raw_rotate/rotate_" + str(angle)
                img_path_list = list_files_recur(angle_group, 'png')[0]
                print(angle_group)
                for path in img_path_list[0:1]:
                    img = cv.imread(path)
                    top_left, match_time, resize_time_1, resize_time_2, optimal_score = \
                        find_reference_point(img, template, n, method_name)
                    image_with_roi = add_roi_rect(img, top_left, template.shape, linewidth=12)

                    save_path = os.path.join(home_folder,  f"{angle}_" + ".png")

                    cv.imwrite(save_path, image_with_roi)
