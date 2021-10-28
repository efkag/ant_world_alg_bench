import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def nans_imgshow(img):
    if np.nanmax(img) > 1.:
        img = img/np.nanmax(img)
    f = plt.figure()
    ax = f.add_subplot(111)
    masked_array = np.ma.array(img, mask=np.isnan(img))
    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad('white', 1.)
    ax.imshow(masked_array, interpolation='nearest', cmap=cmap)
    plt.show()


def plot_3d(data, show=True, rows_cols_idx=111, title=''):
    '''
    Plots the 2d data given in a 3d wireframe.
    Assumes first dimension is number of images,
    second dimension is search angle.
    :param data:
    :return:
    '''
    # The second dimension of the data is the search angle
    # i.e the degree rotated to the left (-deg) and degree rotated to the right (+deg)
    deg = round(data.shape[1]/2)
    no_of_imgs = data.shape[0]

    x = np.linspace(-deg, deg, deg*2)
    y = np.linspace(0, no_of_imgs, no_of_imgs)
    X, Y = np.meshgrid(x, y)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.subplot(rows_cols_idx, projection='3d')
    ax.plot_wireframe(X, Y, data)
    ax.title.set_text(title)
    if show: plt.show()


def plot_multiline(data, scatter=False, labels=None, xlabel=None, ylabel=None):
    if data.ndim < 2:
        data = np.expand_dims(data, axis=0)

    deg = round(data.shape[1] / 2)
    # no_of_imgs = data.shape[0]
    x = np.linspace(-deg, deg, deg * 2)
    for i, line in enumerate(data):
        plt.plot(x, line, label=labels[i])
        if scatter: plt.scatter(x, line)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend()
    plt.show()
