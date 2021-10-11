import sys
import os
path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
# from source.display import nans_imgshow, plot_multiline, plot_3d
from source.analysis import rgb02nan, nanrgb2grey, nanrbg2greyweighted
from source.utils import nanmae, nan_cor_dist, rmf, cor_dist, save_image, rotate
import pickle

os.mkdir(fwd + '/results2')

df = pd.read_csv(fwd + '/office/training.csv')
testdf = pd.read_csv(fwd + '/office/testing.csv')

route = df.to_dict('list')

test = testdf.to_dict('list')
print(test.keys())
snaps = []
for imgfile in route[' Filename']:
    imgfile = imgfile.replace(" ", "")
    print(imgfile)
    img = cv.imread(fwd + '/' + imgfile)
    snaps.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

testimgs = []
for imgfile in test[' Filename']:
    imgfile = imgfile.replace(" ", "")
    img = cv.imread(fwd + '/office/' + imgfile)
    testimgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))


with open(fwd + '/odk_analysis_data_cc.pickle', "rb") as handler:
    datacc = pickle.load(handler)
with open(fwd + '/odk_analysis_data.pickle', "rb") as handler:
    datamae = pickle.load(handler)
print(datamae.keys())

print(datamae['best_index'][35])
print(datacc['best_index'][35])


# set mask pixels to nan
snaps = rgb02nan(snaps)
testimgs = rgb02nan(testimgs)

# a = snaps[0]
# nans_imgshow(a)
# convert to greyscale
greysnaps = nanrgb2grey(snaps)
testimgs = nanrgb2grey(testimgs)

testimg = testimgs[35]

save_image('image-analysis/train.png', img)
save_image('image-analysis/test.png', testimg)


h1 = datamae['heading'][35]
print('heading by mae:', h1)
save_image('image-analysis/test-rotated-mae.png', rotate(h1, testimg))


heat = np.abs(testimgs[35] - greysnaps[127])
save_image('image-analysis/mae-heat.png', heat)


h2 = datacc['heading'][35]
print('heading by cc:', h2)
save_image('image-analysis/test-rotated-cc.png', rotate(h2, testimg))
# plt.imshow(heat, cmap='hot', interpolation='nearest')
# plt.show()



index_mae = datamae['best_index'][35]
index_cc = datacc['best_index'][35]
index_mae = index_cc

mae_sims = []
cc_sims = []
search_angle = 90
half_angle = int(search_angle/2)
deg_range = (-half_angle, half_angle)
degrees = np.arange(*deg_range)
for i, r in enumerate(degrees):

    fig = plt.figure(figsize=(20, 15))
    plt.suptitle('deg={}'.format(r))
    rows = 5
    cols = 2
    ax = fig.add_subplot(rows, cols, 1)
    ax.set_title('train')
    plt.imshow(greysnaps[index_mae], cmap='gray')
    plt.axis('off')

    cols = 2
    ax = fig.add_subplot(rows, cols, 2)
    ax.set_title('train')
    plt.imshow(greysnaps[index_cc], cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 3)
    ax.set_title('test')
    plt.imshow(testimg, cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 4)
    ax.set_title('test')
    plt.imshow(testimg, cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 5)
    abs_diff = nanmae(rotate(r, testimgs[35]), greysnaps[index_mae])
    ax.set_title('test-rotated-mae, {}'.format(abs_diff))
    plt.imshow(rotate(r, testimg), cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 6)
    ccdist = nan_cor_dist(rotate(r, testimgs[35]), greysnaps[index_cc])
    ax.set_title('test-rotated-cc, {}'.format(ccdist))
    plt.imshow(rotate(r, testimg), cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 7)
    heat = np.abs(rotate(r, testimgs[35]) - greysnaps[index_mae])
    mean = np.nanmean(heat)
    # threshold by the mean
    # heat[heat <= mean] = 0
    plt.imshow(heat, cmap='hot', interpolation='nearest')
    ax.set_title('mean abs diff={}'.format(mean))
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 8)
    rotated_img = rotate(r, testimgs[35])
    heat = (rotated_img - np.nanmean(rotated_img)) * (greysnaps[index_cc] - np.nanmean(greysnaps[index_cc]))
    heat[heat <= np.nanmean(heat)] = 0
    cc = nan_cor_dist(rotated_img, greysnaps[index_cc])
    plt.imshow(heat, cmap='hot', interpolation='nearest')
    ax.set_title('cc={}'.format(cc))
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 9)
    ax.set_title('RMAE')
    mae_sims.append(nanmae(greysnaps[index_mae], rotate(r, testimgs[35])))
    plt.plot(degrees[:i+1], mae_sims)
    plt.xlim(-half_angle, half_angle)

    ax = fig.add_subplot(rows, cols, 10)
    ax.set_title('RCC')
    cc_sims.append(nan_cor_dist(greysnaps[index_cc], rotate(r, testimgs[35])))
    plt.plot(degrees[:i+1], cc_sims)
    plt.xlim(-half_angle, half_angle)

    plt.tight_layout(pad=0)
    fig.savefig(fwd + '/results2/' + '{}.png'.format(r+half_angle))
    plt.close(fig)
    # plt.show()
