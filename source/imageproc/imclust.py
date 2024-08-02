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
    '''
    im: Image array in BGR. Assumes the image array is in BRG format
    because the cetroid for the sky is calculated based on the the
    maximus blue value.
    '''
    h, w, c = im.shape
    im_data = im_to_long_form(im)
    im_data = im_data.astype(np.float32, copy=False)
    if whiten: im_data = whiten(im_data)
    # assumes BGR format!!
    cent_sky = im_data[im_data[:, 0].argmax()]
    centroids = [cent_sky, im_data[int(-h/2)]]
    centroids, labels = kmeans2(im_data, k=centroids)
    im_quant = labels.reshape(h, w)
    return im_quant


def extract_skyline(im: np.ndarray):
    skyline = np.argmax(im, axis=0)
    return skyline.max()-skyline

