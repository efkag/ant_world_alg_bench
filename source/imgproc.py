import cv2 as cv
import numpy as np
from pyDTW import DTW

# I have found that this kernel shape works well in general
# In the future I might need to make this into a variable as well
kernel_shape = (5, 5)


def resize(shape):
    '''
    Return a function to resize image to given size
    '''
    return lambda im: cv.resize(im, shape, interpolation=cv.INTER_NEAREST)


def gauss_blur(kernel_shape, mean):
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
    return lambda im: dtw.dtw_haar(im)[-1]


def pipeline(sets):
    '''
    Create a pre-processing pipeline from a dictionary of settings
    :param sets:
    :return:
    '''
    pipe = []
    if sets.get('shape'):
        pipe.append(resize(sets['shape']))
    if sets.get('blur'):
        pipe.append(gauss_blur(kernel_shape, 0))
    if sets.get('edge_range'):
        lims = sets['edge_range']
        pipe.append(canny(lims[0], lims[1]))
    if sets.get('wave'):
        im_size = sets.get('wave')
        pipe.append(wavelet(im_size))
    return pipe
