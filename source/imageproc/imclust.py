import numpy as np
from scipy.cluster.vq import kmeans2, whiten


def im_to_long_form(im: np.ndarray):
    h, w, c = im.shape
    #r, g, b = np.dsplit(im, im.shape[-1])
    im_long = im.reshape(h*w,c)
    ri,ci = np.indices(im.shape[:2])
    #return im_long
    return np.column_stack((im_long, ri.ravel()))


def cluster_im(im: np.ndarray, whiten=False):
    h, w, c = im.shape
    im_data = im_to_long_form(im)
    im_data = im_data.astype(np.float32, copy=False)
    if whiten: im_data = whiten(im_data)
    centroids = [im_data[int(h/2)], im_data[int(-h/2)]]
    centroids, labels = kmeans2(im_data, k=centroids)
    im_quant = labels.reshape(h, w)
    return im_quant


def extract_skyline(im: np.ndarray):
    skyline = np.argmax(im, axis=0)
    return skyline.max()-skyline

