import cv2 as cv
import numpy as np

# I have found that this kernel shape works well in general
# In the future I might need to make this into a variable as well
kernel_shape = (5, 5)


def resize(shape):
    '''
    Return a function to resize image to given size
    '''
    return lambda im: cv.resize(im, shape)


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
    return pipe
