import numpy as np
from scipy.cluster.vq import kmeans2, whiten
import cv2 as cv


def im_to_long_form(im: np.ndarray):
    h, w, c = im.shape
    #r, g, b = np.dsplit(im, im.shape[-1])
    im_long = im.reshape(h*w,c)
    ri,ci = np.indices(im.shape[:2])
    #return im_long
    return np.column_stack((im_long, ri.ravel()))


def cluster_im(im: np.ndarray, whiten_data=False, dialation=False):
    '''
    im: Image array in BGR. Assumes the image array is in BRG format
    because the cetroid for the sky is calculated based on the the
    maximum blue value.
    '''
    h, w, c = im.shape
    im_data = im_to_long_form(im)
    im_data = im_data.astype(np.float32, copy=False)
    #im_data = scale021(im_data)
    if whiten_data: im_data = whiten(im_data)
    # assumes BGR format!!
    cent_sky = im_data[im_data[:, 0].argmax()]
    centroids = [cent_sky, im_data[int(-h/2)]]
    centroids, labels = kmeans2(im_data, k=centroids)
    im_quant = labels.reshape(h, w)

    #dialation
    if dialation:
        im_quant = im_quant.astype(np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        im_quant = cv.dilate(im_quant, kernel, iterations=1)
        im_quant = im_quant.astype()

    return im_quant

def scale021(im: np.ndarray):
    return (im - im.min(axis=0)) / (im.max(axis=0) - im.min(axis=0))

def extract_skyline(im: np.ndarray):
    skyline = np.argmax(im, axis=0)
    return skyline.max()-skyline

