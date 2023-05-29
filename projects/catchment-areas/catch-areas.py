import sys
import os

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from source.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rmse,  rotate, check_for_dir_and_create
from source.routedatabase import Route, BoBRoute
from source.unwraper import Unwraper
from source.display import plot_3d

route_path = 'test-routes/FTLroutes/N-1-01'

route = BoBRoute(path=route_path, read_imgs=True, unwraper=Unwraper)

imgs = route.get_imgs()

params = {'blur': True,
        'shape': (180, 80), 
        #'edge_range': (180, 200)
        }
pipe = Pipeline(**params)
imgs = pipe.apply(imgs)



qi = int(len(imgs)/2)
qimg = imgs[qi]
margin = 50
ref_imgs = imgs[qi-margin:qi+margin]


def catch_areas(query_img, ref_imgs, matcher=mae):
    '''
    Find the catchment areas for each RIDF between the query image and the ref images.
    '''
    ridf_field = rmf(query_img, ref_imgs, matcher=matcher, d_range=(-180, 180))
    indices = np.argmin(ridf_field, axis=1)
    #grad = np.gradient(ridf_field, axis=1)
    diffs = np.diff(ridf_field, axis=1)
    areas = np.empty(len(ref_imgs))
    area_lims = []
    for i, j in enumerate(indices):
        halfright = diffs[i, j:]
        halfleft = diffs[i, :j]
        #find the poit where the gradiend sign change moving away from the minima
        right_lim = j + np.argmax(halfright < 0.0)
        # flip the half left side of the RIDF in order to find the first
        # possitive change fo the gradient moving or the minima to the left
        left_lim = j - np.argmax(np.flip(halfleft) > 0.0)
        area_lims.append((left_lim, right_lim))
        areas[i] = right_lim - left_lim
    return ridf_field, areas, area_lims


def trans_catch_areas(query_img, ref_imgs, matcher=mae):
        '''
        Find the translational catchment areas for each RIDF between the query image and the ref images.
        '''
        ridf_field = rmf(query_img, ref_imgs, matcher=matcher, d_range=(-180, 180))
        #TODO here i need to pick images/RIDF where their minima is less than 45 form the query image.

        # translational idf of the ridf field minima
        tidf = np.min(ridf_field, axis=1)
        # the minima in translation
        min_tidf_i = np.argmin(tidf)
        diffs = np.diff(tidf)

        halfright = diffs[min_tidf_i:]
        halfleft = diffs[:min_tidf_i]
        #find the poit where the gradiend sign change moving away from the minima
        right_lim = min_tidf_i + np.argmax(halfright < 0.0)
        # flip the half left side of the RIDF in order to find the first
        # possitive change fo the gradient moving or the minima to the left
        left_lim = min_tidf_i - np.argmax(np.flip(halfleft) > 0.0)
        area_lims = (left_lim, right_lim)
        area = right_lim - left_lim

        plt.plot(tidf)
        left_lim = area_lims[0]
        right_lim = area_lims[1]
        plt.scatter(range(left_lim, right_lim), tidf[left_lim:right_lim])
        plt.show()

        return ridf_field, area, area_lims


field, area, area_lims = trans_catch_areas(qimg, ref_imgs)
print(area)


#ploting of RIDF catchment areas
# ridfs, areas_size, area_lims =  catch_areas(qimg, ref_imgs)
# for i, ridf in enumerate(ridfs):
#         plt.plot(ridf)
#         left_lim = area_lims[i][0]
#         right_lim = area_lims[i][1]
#         # plt.plot(diffs[i])
#         plt.scatter(range(left_lim, right_lim), ridf[left_lim:right_lim])
#         plt.show()


# check the stat of the area
areas = catch_areas(qimg, ref_imgs)[1]
plt.hist(areas)
plt.show()

plt.boxplot(areas)
plt.show()