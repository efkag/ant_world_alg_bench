import sys
import os

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from source.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rmse, center_ridf, check_for_dir_and_create
from source.routedatabase import Route, BoBRoute
from source.unwraper import Unwraper
from source.display import plot_3d




def catch_areas(query_img, ref_imgs, matcher=mae, **kwargs):
    '''
    Find the catchment areas for each RIDF between the query image and the ref images.
    '''
    ridf_field = rmf(query_img, ref_imgs, matcher=matcher, d_range=(-180, 180))
    ridf_field = center_ridf(ridf_field)
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


def trans_catch_areas(query_img, ref_imgs, matcher=mae, **kwargs):
        '''
        Find the translational catchment areas for each RIDF between the query image and the ref images.
        '''
        ridf_field = rmf(query_img, ref_imgs, matcher=matcher, d_range=(-180, 180))

        # translational idf of the ridf field minima
        tidf = np.min(ridf_field, axis=1)
        # the minima in translation
        min_tidf_i = np.argmin(tidf)
        diffs = np.diff(tidf)

        halfright = diffs[min_tidf_i:]
        halfleft = diffs[:min_tidf_i]
        #find the poit where the gradiend sign change moving away from the minima
        # add 1 cause the diff array is one elment shorter than the tidf
        right_lim = min_tidf_i + np.argmax(halfright < 0.0) + 1
        # flip the half left side of the RIDF in order to find the first
        # possitive change fo the gradient moving or the minima to the left
        left_lim = min_tidf_i - np.argmax(np.flip(halfleft) > 0.0)
        area_lims = (left_lim, right_lim)
        area = right_lim - left_lim

        return ridf_field, area, area_lims


def catch_areas_4route(route, index_step=10, in_translation=False, start_i=10, **kwargs):
    # choose evaluator
    if in_translation:
         evaluator = trans_catch_areas
    else:
         evaluator = catch_areas
    
    imgs = route.get_imgs()
    route_id = route.get_route_id()
    save_path = os.path.join(fwd, f'route{route_id}-results')
    logs = {'route_id':[], 'area':[], 'area_lims':[]}
    check_for_dir_and_create(save_path)
    arrays_save_path = os.path.join(save_path, 'arrays')
    check_for_dir_and_create(arrays_save_path)
    for i in range(start_i, len(imgs), index_step):
         ridf, area, area_lims = evaluator(imgs[i], imgs, **kwargs)
         file = os.path.join(arrays_save_path,f'index:{i}_route:{route_id}')
         logs['area'].append(area.tolist())
         logs['route_id'].append(route_id)
         logs['area_lims'].append(area_lims)
         np.save(file, ridf)
    df = pd.DataFrame(logs)
    file_path = os.path.join(save_path, 'results.csv')
    df.to_csv(file_path, index=False)


# route_path = 'test-routes/FTLroutes/N-1-01'

# route = BoBRoute(path=route_path, read_imgs=True, unwraper=Unwraper)

# imgs = route.get_imgs()

# params = {'blur': True,
#         'shape': (180, 80), 
#         #'edge_range': (180, 200)
#         }
# pipe = Pipeline(**params)
# imgs = pipe.apply(imgs)


# qi = int(len(imgs)/2)
# qimg = imgs[qi]
# margin = 40
# ref_imgs = imgs[qi-margin:qi+margin]
# field, area, area_lims = trans_catch_areas(qimg, ref_imgs)
# print(area)


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
# areas = catch_areas(qimg, ref_imgs)[1]
# plt.hist(areas)
# plt.show()

# plt.boxplot(areas)
# plt.show()