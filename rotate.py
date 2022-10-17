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

# for angle in angles:
#     path = "C:/Temp/roi_study/blister_front/normal_raw_rotate/rotate_" + str(angle)
#     os.makedirs(path, exist_ok=True)
#
#     for idx, img_path in enumerate(good_normal_images[0:25]):
#         img = Image.open(img_path)
#         rotate_img = img.rotate(angle)
#         output_path = os.path.join(path, Path(img_path).stem + "_" + str(angle) + ".png")
#         rotate_img.save(output_path)

###################################################################################################################


method_names = ['cv.TM_CCOEFF_NORMED']
resizing_factors = [1]
roi_templates = list_files_recur(template_folder)[0][0:1]

# rotation_groups = ["C:/Temp/roi_study/blister_front/normal_raw_rotate/rotate_" + str(angle) for angle in angles]



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
            for path in good_normal_images:
                img = cv.imread(path)
                _, _, _, _, optimal_score = find_reference_point(img, template, n, method_name)

                # save information
                row = {
                    'method': method_name,
                    'n': n,
                    'angle': 0,
                    'template': Path(template_path).stem,
                    'image': Path(path).stem,
                    'optimal_score': optimal_score
                }
                L.append(row)

            # rotated image folders
            for angle in angles:
                print(angle)
                angle_group = "C:/Temp/roi_study/blister_front/normal_raw_rotate/rotate_" + str(angle)
                img_path_list = list_files_recur(angle_group, 'png')[0]
                print(angle_group)
                for path in img_path_list:
                    img = cv.imread(path)
                    _, _, _, _, optimal_score = find_reference_point(img, template, n, method_name)

                    # save information
                    row = {
                        'method': method_name,
                        'n': n,
                        'angle': angle,
                        'template': Path(template_path).stem,
                        'image': Path(path).stem,
                        'optimal_score': optimal_score
                    }
                    L.append(row)





df = pd.DataFrame(L)
home_folder = "C:/Temp/roi_study/blister_front"
df.to_csv(os.path.join(home_folder, "blister_rotation_summary.csv"), index=False)
