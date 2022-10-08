import cv2 as cv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pylab
import os
from pathlib import Path
import pandas as pd


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


def resize_image(image, n):
    return cv.resize(image, (int(image.shape[1]/n), int(image.shape[0]/n)))


def find_reference_point(img: np.ndarray, template: np.ndarray, n:int = 1, method_name: str = 'cv2.TM_CCOEFF'):
    """ Find the top left point as the reference point """
    if n != 1:
        img = resize_image(img, n)
        template = resize_image(template, n)

    method = eval(method_name)
    w, h = template.shape[0:2]
    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    top_left = [x*n for x in top_left]
    return top_left


def add_roi_rect(img, top_left, shape):
    """ Add a rectangle for ROI """
    h, w = shape[0:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, (0, 0 ,255), 8) #bgr
    return img


# method -> n -> template -> raw

home_folder = "C:/Temp/roi_study_result"

method_names = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

resizing_factors = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

roi_templates = list_files_recur("C:/Temp/roi_study/blister_front/roi_template")[0]
normal_raw_imgs = list_files_recur("C:/Temp/roi_study/blister_front/normal_raw")[0]


resizing_factors = (1, 2)
method_names = ['cv.TM_CCOEFF']


for method_name in method_names:

    os.makedirs(os.path.join(home_folder, method_name), exist_ok=True)

    for n in resizing_factors:

        os.makedirs(os.path.join(home_folder, method_name, str(n)), exist_ok=True)

        for template_idx, template_path in enumerate(roi_templates):

            os.makedirs(os.path.join(home_folder, method_name, str(n), Path(template_path).stem), exist_ok=True)

            template = cv2.imread(template_path)

            for raw_image_idx, raw_image_path in enumerate(normal_raw_imgs):

                img = cv2.imread(raw_image_path)

                top_left = find_reference_point(img, template, n, method_name)

                image_with_roi = add_roi_rect(img, top_left, template.shape)

                save_path = os.path.join(home_folder, method_name, str(n), Path(template_path).stem, Path(raw_image_path).stem + ".png")

                cv2.imwrite(save_path, image_with_roi)



# img = cv2.imread(normal_raw_imgs[2])
# template = cv2.imread(roi_templates[0])
#
# img.shape
# template.shape
# top_left = find_reference_point(img, template, n=1, method_name='cv2.TM_CCOEFF')
#
# print(top_left)
#



# if __name__ == "__main__":
#
#     print(roi_templates)

# file_path = r"C:\Temp\sample\022584b9-02b3-4e8d-ba1f-3a70b985f656.bmp"
#
# img = cv.imread(file_path, 0)
# img_rgb = cv.imread(file_path)
#
# img2 = img.copy()
#
# template = cv.imread(file_path,0)[600:1600, 90:2350]
#
# w, h = template.shape[::-1]
#
# plt.imshow(template, cmap = 'gray')
# pylab.show()
