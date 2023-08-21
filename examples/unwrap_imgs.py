import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import cv2 as cv
from source.unwraper import Unwraper
from source.imgproc import resize
from source.utils import check_for_dir_and_create


data_path = 'ftl/ftl-live-tests/test_route1'
unwraped_im_path = os.path.join(data_path, 'unwraped_imgs')
check_for_dir_and_create(unwraped_im_path)

data = pd.read_csv(os.path.join(data_path,'database_entries.csv'))

img_files = data[' Filename']

imgs = []
for i in img_files:
    #careful the filenames contain a leading space
    im_path = os.path.join(data_path, i.strip())
    img = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
    imgs.append(img)


unwraper = Unwraper(imgs[0])
resizer = resize((720, 180))
for i, im in enumerate(imgs):
    im = unwraper.unwarp(im)
    im = resizer(im)
    imgs[i] = im


# for i, im in enumerate(imgs):
#     im_file = os.path.join(unwraped_im_path, f'{i}.jpg')
#     cv.imwrite(im_file, im)

#skip frames 
for i in range(0, len(imgs), 2):
    im = imgs[i]
    im_file = os.path.join(unwraped_im_path, f'{i}.jpg')
    cv.imwrite(im_file, im)
