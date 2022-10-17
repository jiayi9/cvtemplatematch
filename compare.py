from main import list_files_recur, resize_image, find_reference_point, add_roi_rect
import cv2 as cv
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

#Blister from raw blister
home_folder = "C:/Temp/roi_study_result/blister_contrast"
template_folder = "C:/Temp/roi_study/blister_front/roi_template"
good_image_folder = "C:/Temp/roi_study/blister_front/normal_raw"
bad_back_image_folder = "C:/Temp/roi_study/blister_front/bad/back"
bad_front_image_folder = "C:/Temp/roi_study/blister_front/bad/front"

resizing_factors = (1, 2, 4, 8, 16, 32, 64)
method_names = ['cv.TM_CCOEFF_NORMED']

roi_templates = list_files_recur(template_folder)[0][0:2]
good_normal_images = list_files_recur(good_image_folder)[0][0:25]
bad_back_images = list_files_recur(bad_back_image_folder)[0][0:25]
bad_front_images = list_files_recur(bad_front_image_folder)[0]

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

            print(1.1)
            # group normal
            for idx, path in enumerate(tqdm(good_normal_images)):
                img = cv.imread(path)
                _, _, _, _, optimal_score = find_reference_point(img, template, n, method_name)

                # save information
                row = {
                    'method': method_name,
                    'n': n,
                    'group': 'normal_images',
                    'template': Path(template_path).stem,
                    'image': Path(path).stem,
                    'optimal_score': optimal_score
                }
                L.append(row)

            print(1.2)

            # group bad back
            for idx, path in enumerate(tqdm(bad_back_images)):
                img = cv.imread(path)
                img = np.concatenate([img, img], axis=0)
                img = np.concatenate([img, img], axis=1)
                _, _, _, _, optimal_score = find_reference_point(img, template, n, method_name)

                # save information
                row = {
                    'method': method_name,
                    'n': n,
                    'group': 'bad_back_images',
                    'template': Path(template_path).stem,
                    'image': Path(path).stem,
                    'optimal_score': optimal_score
                }
                L.append(row)

            print(1.3)

            # group bad back
            for idx, path in enumerate(tqdm(bad_front_images)):
                img = cv.imread(path)
                img = np.concatenate([img, img], axis=0)
                img = np.concatenate([img, img], axis=1)
                _, _, _, _, optimal_score = find_reference_point(img, template, n, method_name)

                # save information
                row = {
                    'method': method_name,
                    'n': n,
                    'group': 'bad_front_images',
                    'template': Path(template_path).stem,
                    'image': Path(path).stem,
                    'optimal_score': optimal_score
                }
                L.append(row)


df = pd.DataFrame(L)
df.to_csv(os.path.join(home_folder, "summary.csv"), index=False)
