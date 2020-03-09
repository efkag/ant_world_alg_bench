import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2 as cv
import math
import os


def display_image(image, size=(10, 10), save_id=None):
    """
    Display the image given as a 2d or 3d array of values.
    :param size: Size of the plot for the image
    :param image: Input image to display
    """
    image = np.squeeze(image)
    fig = plt.figure(figsize=size)
    plt.imshow(image, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # or plt.axis('off')
    if save_id: plt.savefig(str(save_id) + ".png")


def save_image(img, path):
    mpimg.imsave(path + '.png', img, cmap='gray')


def plot_map(world, route_cords=None, grid_cords=None, size=(10, 10), save=False, zoom=(), zoom_factor=1000,
             vectors=None, grid_vectors=None, marker_size=10, scale=40, route_zoom=False, save_id=None, window=None,
             path='', show=True):
    '''
    Plots a top down view of the grid world, with markers or quivers of route and grid positions
    :param world: Top down image of the world
    :param route_cords: X and Y route coordinates
    :param grid_cords: X and Y grid coordinates
    :param size: size of the figure
    :param save: If to save the image
    :param zoom: x and Y tuple of zoom centre
    :param zoom_factor: Magnitute of zoom. (lower values is greater zoom)
    :param vectors: X and Y coordinates of route vectors
    :param grid_vectors: X and Y coordinates of grid vectors
    :param marker_size: Size of route of grid marker
    :param scale: Size of quiver scale. (relative to image size)
    :param route_zoom: Rectangle zoom around the route
    :param save_id: A file id to save the plot and avoid override of the saved file
    :param window: The pointer to the memory window
    :return: -
    '''
    fig = plt.figure(figsize=size)
    fig.suptitle('World Grid', fontsize=16, fontweight='bold')
    plt.xlabel('x coordinates', fontsize=14, fontweight='bold')
    plt.ylabel('y coordinates', fontsize=13, fontweight='bold')
    # Plot circles for route image locations
    if route_cords and not vectors: plt.scatter(route_cords[0], route_cords[1], marker="o", s=marker_size, color='blue')
    # Plot stars for grid image locations
    if grid_cords and not grid_vectors: plt.scatter(grid_cords[0][0:save_id], grid_cords[1][0:save_id], marker="*", s=marker_size,
                                                    color='red')
    # Plot route images heading vectors
    if vectors: plt.quiver(route_cords[0], route_cords[1], vectors[0], vectors[1], scale=scale, color='b')
    # Plot world grid images heading vectors
    # if grid_vectors: plt.quiver(grid_cords[0], grid_cords[1], grid_vectors[0], grid_vectors[1], scale=scale, color='r')
    # The variable save_id is used here to plot the vectors that have been matched with a window so far
    if grid_vectors: plt.quiver(grid_cords[0][0:save_id], grid_cords[1][0:save_id],
                                grid_vectors[0][0:save_id], grid_vectors[1][0:save_id], scale=scale, color='r')
    # Plot window vectors only
    if window:
        window = range(window[0], window[1])
        plt.quiver([route_cords[0][i] for i in window], [route_cords[1][i] for i in window],
                   [vectors[0][i] for i in window], [vectors[1][i] for i in window], scale=scale, color='c')

    plt.imshow(world, zorder=0, extent=[-0.158586 * 1000, 10.2428 * 1000, -0.227704 * 1000, 10.0896 * 1000])
    if zoom:
        plt.xlim([zoom[0] - zoom_factor, zoom[0] + zoom_factor])
        plt.ylim([zoom[1] - zoom_factor, zoom[1] + zoom_factor])
    if route_zoom:
        # plt.xlim([])
        plt.ylim([4700, 6500])
    if save: fig.savefig(path + 'graph' + str(save_id) + '.png')
    if show: plt.show()


def load_grid():
    grid_dir = 'AntWorld/world5000_grid/'

    data = pd.read_csv(grid_dir + 'world5000_grid.csv', header=0)
    data = data.values

    # World top down image
    world = mpimg.imread(grid_dir + 'world5000_grid.png')

    # Grid data
    x = data[:, 1]  # x location of the image in the world_grid
    y = data[:, 0]  # y location of the image in the world_grid
    # img_path = data[:, 4]  # Name of the image file

    return x, y, world


def gen_route_line(indexes, headings, direction, length):
    if direction == 'right':
        index_jump = 105
        head = [0]
    elif direction == 'left':
        index_jump = -105
        head = [180]
    elif direction == 'up':
        index_jump = 1
        head = [90]
    elif direction == 'down':
        index_jump = -1
        head = [270]
    elif direction == 'up_r':
        index_jump = 106
        head = [45]
    elif direction == 'up_l':
        index_jump = -104
        head = [135]
    elif direction == 'down_r':
        index_jump = 104
        head = [315]
    elif direction == 'down_l':
        index_jump = -106
        head = [225]
    else: raise Exception('Wrong direction given')

    # for i in range(length):
    #     indexes.append(indexes[-1] + index_jump)
    # Add indexes using the range
    end = indexes[-1] + length*index_jump + index_jump
    indexes.extend(list(range(indexes[-1]+index_jump, end, index_jump)))
    headings.extend(head * length)

    return indexes, headings


def route_imgs_from_indexes(indexes, headings, directory):
    grid_dir = 'AntWorld/world5000_grid/'
    data = pd.read_csv(grid_dir + 'world5000_grid.csv', header=0)
    data = data.values
    img_path = data[:, 4]  # Name of the image files

    images = []
    id = 0
    for i, h in zip(indexes, headings):
        img = cv.imread(grid_dir + img_path[i][1:], cv.IMREAD_GRAYSCALE)
        img = rotate(h, img)
        save_image(img, directory + str(id))
        images.append(img)
        id += 1

    return images


def load_route(route_id, grid_pos_limit=200):
    # Path/ Directory settings
    route_id_dir = 'ant1_route' + route_id + '/'
    route_dir = 'AntWorld/' + route_id_dir
    grid_dir = 'AntWorld/world5000_grid/'

    # World top down image
    world = mpimg.imread(grid_dir + 'world5000_grid.png')

    # Grid Images
    data = pd.read_csv(grid_dir + 'world5000_grid.csv', header=0)
    data = data.values

    # Route
    route_data = pd.read_csv(route_dir + 'ant1_route' + route_id + '.csv', header=0)
    route_data = route_data.values

    ## Organize data
    # Grid data
    X = data[:, 1]  # x location of the image in the world_grid
    Y = data[:, 0]  # y location of the image in the world_grid
    img_path = data[:, 4]  # Name of the image file

    # Route data
    X_route = route_data[:, 1].tolist()  # x location of the image in the route
    Y_route = route_data[:, 0].tolist()  # y location of the image in the route
    Heading_route = route_data[:, 3]  # Image heading
    imgs_route_path = route_data[:, 4]  # Name of the image file

    # Load route images
    max_norm = 1
    route_images = []
    for i in range(0, len(imgs_route_path)):
        img = cv.imread(route_dir + imgs_route_path[i][1:], cv.IMREAD_GRAYSCALE)
        # Normalize
        #img = img * max_norm / img.max()
        route_images.append(img)

    # Load world grid images
    max_norm = 1
    X_inlimit = []
    Y_inlimit = []
    world_grid_imgs = []

    # Fetch images from the grid that are located nearby route images.
    for i in range(0, len(X), 1):
        dist = []
        for j in range(0, len(X_route), 1):
            d = (math.sqrt((X_route[j] - X[i]) ** 2 + (Y_route[j] - Y[i]) ** 2))
            dist.append(d)
        if min(dist) < grid_pos_limit:  # Maximum distance limit from the Route images
            X_inlimit.append(X[i])
            Y_inlimit.append(Y[i])
            img_dir = grid_dir + img_path[i][1:]
            img = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
            # Normalize
            # img = img * max_norm / img.max()
            world_grid_imgs.append(img)

    return world, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, Heading_route, route_images


def sample_from_wg(x_cords, y_cords, x_route_cords, y_route_cords, world_grid_imgs, min_dist):
    '''
    Samples images and their coordinates from the world grid
    given a routes coordinates and a min distance.
    The min distance would be the distance between
    a grid image and the nearest route image.


    :param x_cords: world grid coordinate
    :param y_cords: world grid coordinate
    :param x_route_cords: x route coordinate
    :param y_route_cords: y route coordinate
    :param world_grid_imgs: Grid images
    :param min_dist: Minimum distance between grid poit and route point.
    :return:
    '''
    x_inrange = []
    y_inrange = []
    w_g_imgs_inrange = []

    for i in range(0, len(x_cords), 1):
        dist = []
        for j in range(0, len(x_route_cords), 1):
            d = (math.sqrt((x_route_cords[j] - x_cords[i]) ** 2 + (y_route_cords[j] - y_cords[i]) ** 2))
            dist.append(d)
        if min(dist) < min_dist:
            x_inrange.append(x_cords[i])
            y_inrange.append(y_cords[i])
            w_g_imgs_inrange.append(world_grid_imgs[i])

    x_inrange = list(reversed(x_inrange))
    y_inrange = list(reversed(y_inrange))
    w_g_imgs_inrange = list(reversed(w_g_imgs_inrange))

    return x_inrange, y_inrange, w_g_imgs_inrange


def line_incl(x, y):
    '''
    Calculates the inclination of lines defined by 2 subsequent coordinates
    :param x:
    :param y:
    :return:
    '''
    incl = np.arctan((y[1:] - y[:-1]) / (x[1:] - x[:-1])) * 180 / np.pi
    return incl


def pol2cart(theta, r):
    """
    This function converts the cartesian coordinates into polar coordinates.
    of the quiver.
    :param theta: represents the heading in degrees
    :param r: represens the lenghth of the quiver
    :return: This function returns a tuple (float, float) which represents the u and v coordinates
    """
    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    return x, y


def pol_2cart_headings(headings):
    """
    Convert degree headings to U,V cartesian coordinates
    :param headings: list of degrees
    :return: 2D coordinates
    """
    U = []
    V = []

    for i in range(0, len(headings)):
        U.append(pol2cart(headings[i], 1)[0])
        V.append(pol2cart(headings[i], 1)[1])

    return U, V


def pre_process(imgs, sets):
    """
    Gaussian blur, edge detection and image resize
    :param imgs:
    :param shape:
    :param edges:
    :return:
    """
    if sets.get('edge_range'):
        lims = sets['edge_range']
        imgs = [cv.Canny(img, lims[0], lims[1]) for img in imgs]
    if sets.get('blur'):
        imgs = [cv.GaussianBlur(img, (5, 5), 0) for img in imgs]
    if sets.get('shape'):
        shape = sets['shape']
        imgs = [cv.resize(img, shape) for img in imgs]
    return imgs


def rotate(d, image):
    """
    Converts the degrees into columns and rotates the image.
    :param d: number of degrees the the agent will rotate its view
    :param image: An np.array that we want to shift.
    :return: Returns the rotated image.
    """
    if abs(d) > 360:
      deg = abs(d) - 360
    if d < 0:
        d = -d
        num_of_cols = image.shape[1]
        num_of_cols_perdegree = num_of_cols/360
        cols_to_shift = round(d * num_of_cols_perdegree)
        img1 = image[:, cols_to_shift:num_of_cols]
        img2 = image[:, 0: cols_to_shift]
        return np.concatenate((img1, img2), axis=1)
    else:
        num_of_cols = image.shape[1]
        num_of_cols_perdegree = num_of_cols / 360
        cols_to_shift = num_of_cols - round(d * num_of_cols_perdegree)
        img1 = image[:, cols_to_shift:num_of_cols]
        img2 = image[:, 0: cols_to_shift]
        return np.concatenate((img1, img2), axis=1)


def idf(img, ref_img):
    """
    Image Differencing Function RMSE
    :param img:
    :param ref_img:
    :return:
    """
    return math.sqrt(((ref_img - img)**2).mean())


def cov(a, b):
    """
    Calculates covariance (non sample)
    Assumes flattened arrays
    :param a:
    :param b:
    :return:
    """
    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    return np.sum((a - a_mean) * (b - b_mean)) / (len(a))


def cor_coef(a, b):
    """
    Calculate correlation coefficient
    :param a:
    :param b:
    :return:
    """
    a = a.flatten()
    b = b.flatten()
    return cov(a, b) / (np.std(a) * np.std(b))


def flatten_imgs(imgs):
    return [img.flatten() for img in imgs]


def cross_corr(sub_series, series):
    return [np.dot(s, sub_series) for s in series]


def degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings):
    k = []  # Holds the position of the memory with the shortest distance to the wg position
    errors = []  # Holds the error between the world grid image and the closest route image
    for i in range(0, len(x_cords)):  # For every grid position
        distance = []
        for j in range(0, len(x_route_cords)):  # For every route position
            d = math.sqrt((x_cords[i] - x_route_cords[j]) ** 2 + ((y_cords[i] - y_route_cords[j]) ** 2))
            distance.append(d)

        k.append(distance.index(min(distance)))
        errors.append(abs(recovered_headings[i] - route_heading[distance.index(min(distance))]))
    return errors


def mean_degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings):
    error = degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings)
    return sum(error) / len(error)


def check_for_dir_and_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_grid_route(route_dir, route_id=1, grid_pos_limit=200, route_direction='left2right'):
    # Path/ Directory settings
    route_id_dir = 'route_' + str(route_id) + '/'
    csv_file = 'route_' + str(route_id) + '.csv'
    route_dir = route_dir + route_id_dir
    grid_dir = 'AntWorld/world5000_grid/'

    # World top down image
    world = mpimg.imread(grid_dir + 'world5000_grid.png')

    # Grid Images
    data = pd.read_csv(grid_dir + 'world5000_grid.csv', header=0)
    data = data.values

    # Route
    route_data = pd.read_csv(route_dir + csv_file, header=0)
    route_data = route_data.values

    ## Organize data
    # Grid data
    X = data[:, 1]  # x location of the image in the world_grid
    Y = data[:, 0]  # y location of the image in the world_grid
    img_path = data[:, 4]  # Name of the image file

    # Route data
    X_route = route_data[:, 0].tolist()  # x location of the image in the route
    Y_route = route_data[:, 1].tolist()  # y location of the image in the route
    Heading_route = route_data[:, 3]  # Image heading
    imgs_route_path = route_data[:, 4]  # Name of the image file

    # Load route images
    max_norm = 1
    route_images = []
    for i in range(0, len(imgs_route_path)):
        img = cv.imread(route_dir + imgs_route_path[i], cv.IMREAD_GRAYSCALE)
        # Normalize
        #img = img * max_norm / img.max()
        route_images.append(img)

    # Load world grid images
    max_norm = 1
    X_inlimit = []
    Y_inlimit = []
    world_grid_imgs = []

    # Fetch images from the grid that are located nearby route images.
    for i in range(0, len(X_route), 1):
        dist = []
        for j in range(0, len(X), 1):
            d = (math.sqrt((X_route[i] - X[j]) ** 2 + (Y_route[i] - Y[j]) ** 2))
            dist.append(d)
            if grid_pos_limit > d > 0.1:  # Maximum distance limit from the Route images
                X_inlimit.append(X[j])
                Y_inlimit.append(Y[j])
                img_dir = grid_dir + img_path[j][1:]
                img = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
                # Normalize
                # img = img * max_norm / img.max()
                world_grid_imgs.append(img)

    if route_direction == 'right2left':
        X_inlimit = list(reversed(X_inlimit))
        Y_inlimit = list(reversed(Y_inlimit))
        world_grid_imgs = list(reversed(world_grid_imgs))
    elif route_direction == 'left2right':
        # If direction is left to right the order of indexes is in the correct order.
        pass
    else:
        raise Exception('Provided wrong route direction parameter')

    return world, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, Heading_route, route_images