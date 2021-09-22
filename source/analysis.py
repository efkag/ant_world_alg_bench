import numpy as np
from scipy.spatial.distance import cdist
import os
import cv2 as cv
from source.utils import rotate
from source import antworld2


def iqr_outliers(data):
    '''
    Calculates the inter-quartile outliers.
    Q3-Q1
    :param data:
    :return:
    '''
    q3 = np.percentile(data, 75)
    q1 = np.percentile(data, 25)
    iqr = q3 - q1
    outliers_over = data[data > (q3 + 1.5 * iqr)]
    outliers_under = data[data < (q1 - 1.5 * iqr)]
    out = np.append(outliers_over, outliers_under)
    return out


def perc_outliers(data):
    '''
    Calculates of the percetage of outlires as
    defined by the iqroutliers function
    :param data:
    :return:
    '''
    out = iqr_outliers(data)
    perc = len(out)/len(data)
    return perc


def log_error_points(route, traj, thresh=0.5, route_id=1, target_path=None):
    if target_path:
        logs_path = os.path.join(target_path, 'route' + str(route_id))
    else:
        logs_path = 'route' + str(route_id)
    os.mkdir(logs_path)
    # the antworld agent
    agent = antworld2.Agent()
    # get xy coords
    traj_xy = np.column_stack(traj['x'], traj['y'])
    route_xy = np.column_stack(route['x'], route['y'])

    for i in range(len(traj['heading'])):
        dist = np.squeeze(cdist(np.expand_dims(traj_xy[i], axis=0), route_xy, 'euclidean'))
        index = np.argmin(dist)
        min_dist = dist[index]
        if min_dist > thresh:
            point_path = os.path.join(logs_path, str(i))
            os.mkdir(point_path)
            # Save window images
            if traj.get('window_log'):
                w = traj.get('window_log')
                for wi in range(w[0], w[1]):
                    cv.imwrite(point_path + route['filename'][wi], route['imgs'][wi])
            # Save the query image
            h = traj['heading'][i]
            img = agent.get_img(traj_xy[i], h)
            rimg = rotate(h, img)
            cv.imwrite(point_path + str(h) + '.png', rimg)

            # TODO: save heatmap for wrsims for the given test position image


def rgb02nan(imgs, color=None):
    if not color:
        color = (0.0, 0.0, 0.0)
    nans = [np.nan, np.nan, np.nan]
    for i, img in enumerate(imgs):
        img = img.astype(np.float64)
        indices = np.where(np.all(img == color, axis=-1))

        for r, c in zip(indices[0], indices[1]):
            img[r, c, :] = nans
        imgs[i] = img
    return imgs


def nanrgb2grey(imgs):
    """
    Turn RGB images with NaNs to greyscale
    :param imgs:
    :return:
    """
    if isinstance(imgs, list):
        return [np.nanmean(img, axis=-1) for img in imgs]

    return np.nanmean(imgs, axis=-1)


def nanrbg2greyweighted(imgs):
    """
    Turn RGB images with NaNs to greyscale with weights for each channel
    :param imgs:
    :return:
    """
    rgb_weights = [0.2989, 0.5870, 0.1140]

    if isinstance(imgs, list):
        return [np.average(img, weights=rgb_weights, axis=-1) for img in imgs]

    return np.average(imgs, weights=rgb_weights, axis=-1)
