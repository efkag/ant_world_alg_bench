import sys
import os

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

from datetime import date
today = date.today()
string_date = today.strftime("%Y-%m-%d")

import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from source.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rmse, squash_deg, travel_dist, angular_diff, center_ridf, check_for_dir_and_create
from source.routedatabase import Route, BoBRoute
from source.unwraper import Unwraper




def rot_catch_areas(query_img, ref_imgs, matcher=mae, deg_range=(-180, 180), **kwargs):
    '''
    Find the catchment areas for each RIDF between the query image and the ref images.
    '''
    degrees = np.arange(*deg_range)
    ridf_field = rmf(query_img, ref_imgs, matcher=matcher, d_range=deg_range)
    #### get the angular errors of the RIDFS
    route = kwargs.get('route')
    yaw = squash_deg(route.get_yaw())
    indices = np.argmin(ridf_field, axis=1)
    headings = np.take(degrees, indices)
    headings = squash_deg(headings + yaw)
    adiffs = angular_diff(headings, yaw)
    #### cneter the RIDF to get the Rotational CA of each later
    ridf_field = center_ridf(ridf_field)
    indices = np.argmin(ridf_field, axis=1)
    #grad = np.gradient(ridf_field, axis=1)
    diffs = np.diff(ridf_field, axis=1)
    areas = np.empty(len(ref_imgs))
    area_lims = []
    # calac the rotCA for each
    for i, j in enumerate(indices):
        halfright = diffs[i, j:]
        halfleft = diffs[i, :j]
        #find the poit where the gradiend sign change moving away from the minima
        right_lim = j + np.argmax(halfright < 0.0) + 1
        # flip the half left side of the RIDF in order to find the first
        # possitive change fo the gradient moving or the minima to the left
        left_lim = j - np.argmax(np.flip(halfleft) > 0.0)
        area_lims.append((left_lim, right_lim))
        areas[i] = right_lim - left_lim
    return ridf_field, areas, area_lims, adiffs


def trans_catch_areas(query_img, ref_imgs, matcher=mae, error_thresh=25, 
                      deg_range=(-180, 180), **kwargs):
        '''
        Find the translational catchment areas for each RIDF between the query image and the ref images.
        '''
        ridf_field = rmf(query_img, ref_imgs, matcher=matcher, d_range=deg_range)
        degrees = np.arange(*deg_range)
        ref_i = kwargs.get('ref_i')
        adiffs = np.array([])
        if error_thresh:
            route = kwargs.get('route')
            yaw = squash_deg(route.get_yaw())
            indices = np.argmin(ridf_field, axis=1)
            headings = np.take(degrees, indices)
            headings = squash_deg(headings + yaw)
            adiffs = angular_diff(headings, yaw)
            if np.argmax(adiffs[ref_i:] > error_thresh).all() == False:
                right_i = ref_i + len(adiffs[ref_i:])
            else:
                right_i = ref_i + np.argmax(adiffs[ref_i:] > error_thresh)
            if np.argmax(np.flip(adiffs[:ref_i]) > error_thresh).all() == False:
                left_i = ref_i - len(adiffs[:ref_i])
            else:
                left_i = ref_i - np.argmax(np.flip(adiffs[:ref_i]) > error_thresh)
            # save the angular diffs of the tCA for analysis later
            adiffs = adiffs[left_i:right_i]
        # This is done to reset the indices that fall within the AAE thresh
        # Previousl;y the indices were used to constrain the catchment areas search.
        # By resseting them we make it the CA serach indepedent of the AAE threshold 
        left_i = 0
        right_i = None
        # translational idf of the ridf field minima
        tidf = np.min(ridf_field, axis=1)
        tidf = tidf[left_i:right_i]
        # the minima in translation
        min_tidf_i = np.argmin(tidf)
        diffs = np.diff(tidf)

        halfright = diffs[min_tidf_i:]
        halfleft = diffs[:min_tidf_i]
        #find the poit where the gradiend sign change moving away from the minima
        # add 1 cause the diff array is one elment shorter than the tidf
        if halfright.size == 0:
            print('here')
        if np.argmax(halfright < 0.0).all() == False:
            right_lim = left_i + min_tidf_i + len(halfright) + 1
        else:
            right_lim = left_i + min_tidf_i + np.argmax(halfright < 0.0) + 1
        # flip the half left side of the RIDF in order to find the first
        # possitive change fo the gradient moving or the minima to the left
        if np.argmax(np.flip(halfleft) > 0.0).all() == False:
            left_lim = left_i + min_tidf_i - len(halfleft)
        else:
            left_lim = left_i + min_tidf_i - np.argmax(np.flip(halfleft) > 0.0)
        
        area_lims = (left_lim, right_lim)
        area = right_lim - left_lim

        return ridf_field, area, area_lims, adiffs


def catch_areas_4route(route, pipe=None, index_step=10, in_translation=False, 
                       start_i=10, **kwargs):
    # choose evaluator
    if in_translation:
         evaluator = trans_catch_areas
    else:
         evaluator = rot_catch_areas
    
    imgs = route.get_imgs()
    xy = route.get_xycoords()
    if pipe:
        imgs = pipe.apply(imgs)
    route_id = route.get_route_id()
    save_path = os.path.join(fwd, string_date, f'route{route_id}-results')
    logs = {'route_id':[],'ref_i':[], 'area':[], 'area_lims':[], 'area_cm':[], 'adiffs':[]}
    check_for_dir_and_create(save_path)
    arrays_save_path = os.path.join(save_path, 'arrays')
    check_for_dir_and_create(arrays_save_path)
    for i in range(start_i, len(imgs)-start_i, index_step):
        ridf, area, area_lims, adiffs = evaluator(imgs[i], imgs, route=route, ref_i=i, **kwargs)
        logs['ref_i'].append(i)
        if type(area) is np.ndarray:
            logs['area'].append(area.tolist())
        else:
           logs['area'].append(area)
        logs['route_id'].append(route_id)
        logs['area_lims'].append(area_lims)
        logs['adiffs'].append(adiffs.tolist())
        if in_translation:
            x = xy['x'][area_lims[0]:area_lims[1]]
            y = xy['y'][area_lims[0]:area_lims[1]]
            ### This depends on the data base
            d_cm = travel_dist(x, y) / 10
            logs['area_cm'].append(d_cm)
        file = os.path.join(arrays_save_path,f'index:{i}_route:{route_id}')
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