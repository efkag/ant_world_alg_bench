import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from source.display import nans_imgshow, plot_ridf_multiline, plot_3d
from source.analysis import rgb02nan, nanrgb2grey, nanrbg2greyweighted
from source.utils import nanmae, nan_cor_dist, rmf, cor_dist, save_image, rotate
import time

df = pd.read_csv('office/training.csv')
testdf = pd.read_csv('office/testing.csv')

route = df.to_dict('list')

test = testdf.to_dict('list')
print(test.keys())
snaps = []
for imgfile in route[' Filename']:
    imgfile = imgfile.replace(" ", "")
    img = cv.imread(imgfile)
    snaps.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

testimgs = []
for imgfile in test[' Filename']:
    imgfile = imgfile.replace(" ", "")
    img = cv.imread('office/' + imgfile)
    testimgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

snaps = rgb02nan(snaps)
testimgs = rgb02nan(testimgs)

greysnaps = nanrgb2grey(snaps)
testimgs = nanrgb2grey(testimgs)

timesteps = 10000
timelog = []
for i in range(timesteps):
    tic = time.perf_counter()
    nanmae(testimgs[0], greysnaps[0])
    toc = time.perf_counter()
    timelog.append(toc-tic)

mae_time = np.mean(timelog)
print(mae_time)

timelog = []
for i in range(timesteps):
    tic = time.perf_counter()
    nan_cor_dist(testimgs[0], greysnaps[0])
    toc = time.perf_counter()
    timelog.append(toc-tic)

cc_time = np.mean(timelog)
print(cc_time)

print(cc_time/mae_time)
