import cv2 as cv
import numpy as np
from source.pyDTW import DTW



def resize(shape):
    '''
    Return a function to resize image to given size
    '''
    return lambda im: cv.resize(im, shape, interpolation=cv.INTER_NEAREST)


def gauss_blur(mean, kernel_shape=(5, 5)):
    '''
    Return a function to blur image
    given the kernel size and the mean
    '''
    return lambda im: cv.GaussianBlur(im, kernel_shape, mean)


def canny(upper, lower):
    '''
    Return a function to perform Canny edge detection
    within the given thresholds
    '''
    return lambda im: cv.Canny(im, upper, lower)


def standardize():
    return lambda im: (im - np.mean(im)) / np.std(im)


def wavelet(image_shape):
    dtw = DTW(image_shape)
    return lambda im: dtw.dwt_haar(im)[-1]


def mod_dtype(dtype):
    return lambda im: im.astype(dtype)


def lin(img, kernel_shape=(3, 3)):
    '''
    Local Image Normalisation
    Normalises and standarises each pixel using 
    the mean the std.dev of the neighboring pixels
    '''
    img = img.astype(np.int16, copy=False)
    mu = cv.blur(img, kernel_shape)
    img = img - mu
    var = cv.blur(img*img, kernel_shape)
    sig = var**0.5 #+ np.finfo(float).eps
    return img / sig


def loc_norm(kernel_shape=(3, 3)):
    return lambda im: lin(im, kernel_shape)


def glin(img, sig1=2, sig2=20):
    '''
    Gaussian Local Image Normalisation
    Normalises and standarises each pixel using 
    the weighted mean the std.dev of the neighboring pixels
    The weighted mean is calculated using the gaussian kernel
    '''
    img = img.astype(np.int16, copy=False)
    mu = cv.GaussianBlur(img, (0, 0), sig1)
    img = img - mu
    var = cv.GaussianBlur(img*img, (0, 0), sig2)
    sig = var**0.5 #+ np.finfo(float).eps
    return img / sig


def gauss_loc_norm(sig1=2, sig2=20):
    return lambda im: glin(im, sig1, sig2)


def imvcrop(shape=None, vcrop=None):
    if vcrop < 1. and vcrop > 0:
        vcrop = int(round(shape[1] * vcrop))
    else:
        vcrop = None
    return lambda im: im[vcrop:, :]


def histeq():
    return lambda im: cv.equalizeHist(im)


def make_pipeline(sets):
    '''
    Create a pre-processing pipeline from a dictionary of settings
    :param sets:
    :return:
    '''
    pipe = []
    if sets.get('shape'):
        pipe.append(resize(sets['shape']))
    if sets.get('vcrop'):
        im_shape = sets.get('shape')
        pipe.append(imvcrop(shape=im_shape, vcrop=sets.get('vcrop')))
    if sets.get('histeq'):
        pipe.append(histeq())
    if sets.get('blur'):
        pipe.append(gauss_blur(0))
    if sets.get('edge_range'):
        lims = sets['edge_range']
        pipe.append(canny(lims[0], lims[1])) 
    if sets.get('loc_norm'):
        pipe.append(loc_norm(**sets.get('loc_norm')))
    if sets.get('gauss_loc_norm'):
        pipe.append(gauss_loc_norm(**sets.get('gauss_loc_norm')))
    if sets.get('wave'):
        im_size = sets.get('shape')
        pipe.append(wavelet(im_size))
    # Always change the datatype to int16 or float32 to avoid wrap-around!!
    pipe.append(mod_dtype(np.float32))
    return pipe


class Pipeline:
    def __init__(self, **sets) -> None:
        if sets:
            self.pipe = make_pipeline(sets)
        else:
            self.pipe = []
            self.pipe.append(mod_dtype(np.float32))

    def apply(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        for p in self.pipe:
            imgs = [p(img) for img in imgs]
        return imgs if len(imgs) > 1 else imgs[0]