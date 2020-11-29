import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import cv2 as cv
import math
import os
sns.set(font_scale=0.8)


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
    plt.title("C", loc="left")
    plt.tight_layout(pad=0)
    if save_id: fig.savefig(str(save_id) + ".png", bbox_inches="tight")
    plt.show()


def display_split(image_l, image_r, size=(10, 10), file_name=None):
    image_l = np.squeeze(image_l)
    image_r = np.squeeze(image_r)

    fig = plt.figure(figsize=size)
    fig.add_subplot(1, 2, 1)
    plt.imshow(image_l, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    fig.add_subplot(1, 2, 2)
    plt.imshow(image_r, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # or plt.axis('off')
    if file_name: plt.savefig(file_name + ".png")
    plt.show()


def save_image(path, img):
    cv.imwrite(path, img)# , cmap='gray')


def plot_map(world, route_cords=None, grid_cords=None, size=(10, 10), save=False, zoom=(), zoom_factor=1500,
             route_headings=None, grid_headings=None, error_indexes=None, marker_size=5, scale=40, route_zoom=False, save_id=None, window=None,
             path='', show=True, title=None):
    '''
    Plots a top down view of the grid world, with markers or quivers of route and grid positions
    :param world: Top down image of the world
    :param route_cords: X and Y route coordinates
    :param grid_cords: X and Y grid coordinates
    :param size: size of the figure
    :param save: If to save the image
    :param zoom: x and y tuple of zoom centre
    :param zoom_factor: Magnitude of zoom. (lower values is greater zoom)
    :param route_headings: Heading in degrees of the route positions
    :param grid_headings: Heading in degrees of grid positions
    :param marker_size: Size of route of grid marker
    :param scale: Size of quiver scale. (relative to image size)
    :param route_zoom: Rectangle zoom around the route
    :param save_id: A file id to save the plot and avoid override of the saved file
    :param window: The pointer to the memory window
    :return: -
    '''
    fig = plt.figure(figsize=size)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.xlabel('x coordinates', fontsize=14, fontweight='bold')
    plt.ylabel('y coordinates', fontsize=13, fontweight='bold')
    # Plot circles for route image locations
    if route_cords and route_headings is None: plt.plot(route_cords[0], route_cords[1],
                                                        marker="o", markersize=marker_size, linewidth=1, color='blue')
    # Plot stars for grid image locations
    if grid_cords and grid_headings is None: plt.scatter(grid_cords[0][0:save_id], grid_cords[1][0:save_id],
                                                         marker="*", s=marker_size, color='red')
    # Plot route images heading vectors
    if route_headings is not None:
        route_U, route_V = pol_2cart_headings(90.0 - np.array(route_headings))
        plt.quiver(route_cords[0], route_cords[1], route_U, route_V, scale=scale, color='b')
    # Plot world grid images heading vectors
    # The variable save_id is used here to plot the vectors that have been matched with a window so far
    if grid_headings is not None and error_indexes is None:
        grid_U, grid_V = pol_2cart_headings(90.0 - np.array(grid_headings))
        plt.quiver(grid_cords[0][0:save_id], grid_cords[1][0:save_id],
                                grid_U[0:save_id], grid_V[0:save_id], scale=scale, color='r')
    if error_indexes:
        grid_U, grid_V = pol_2cart_headings(90.0 - np.array(grid_headings))
        plt.quiver(grid_cords[0], grid_cords[1], grid_U, grid_V, scale=scale, color='b')
        error_headings = [grid_headings[i] for i in error_indexes]
        error_X = [grid_cords[0][i] for i in error_indexes]
        error_Y = [grid_cords[1][i] for i in error_indexes]
        error_U, error_V = pol_2cart_headings(90.0 - np.array(error_headings))
        plt.quiver(error_X, error_Y, error_U, error_V, scale=scale, color='r')

    # Plot window vectors only
    if window:
        window = range(window[0], window[1])
        route_U, route_V = pol_2cart_headings(90.0 - np.array(route_headings))
        plt.quiver([route_cords[0][i] for i in window], [route_cords[1][i] for i in window],
                   [route_U[i] for i in window], [route_V[i] for i in window], scale=scale, color='c')

    plt.imshow(world, zorder=0, extent=[-0.158586 * 1000, 10.2428 * 1000, -0.227704 * 1000, 10.0896 * 1000])
    if zoom:
        plt.xlim([zoom[0] - zoom_factor, zoom[0] + zoom_factor])
        plt.ylim([zoom[1] - zoom_factor, zoom[1] + zoom_factor])
    if route_zoom:
        # plt.ylim([])
        plt.xlim([4700, 6500])
    plt.xticks([]), plt.yticks([])
    # plt.title("A", loc="left", fontsize=20)
    plt.tight_layout(pad=0)
    if save and save_id: fig.savefig(path + 'graph' + str(save_id) + '.png')
    if save and not save_id: fig.savefig(path)
    if show: plt.show()
    if save: plt.close()


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
        save_image(directory + str(id) + '.png', img)
        images.append(img)
        id += 1

    return images


def element_index(l, elem):
    try:
        return l.index(elem)
    except ValueError:
        return False


def load_route(route_id, grid_pos_limit=200):
    # Path/ Directory settings
    antworld_path = '../AntWorld'
    route_id = str(route_id)
    route_id_dir = '/ant1_route' + route_id + '/'
    route_dir = antworld_path + route_id_dir
    grid_dir = antworld_path + '/world5000_grid/'

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
    X = data[:, 0]  # x location of the image in the world_grid
    Y = data[:, 1]  # y location of the image in the world_grid
    img_path = data[:, 4]  # Name of the image file

    # Route data
    X_route = route_data[:, 0].tolist()  # x location of the image in the route
    Y_route = route_data[:, 1].tolist()  # y location of the image in the route
    Heading_route = route_data[:, 3]     # Image heading
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
    for i in range(0, len(X_route)):
        dist = []
        for j in range(0, len(X)):
            d = (math.sqrt((X_route[i] - X[j]) ** 2 + (Y_route[i] - Y[j]) ** 2))
            dist.append(d)
            if grid_pos_limit > d:  # Maximum distance limit from the Route images
                X_inlimit.append(X[j])
                Y_inlimit.append(Y[j])
                X[j] = 0
                Y[j] = 0
                img_dir = grid_dir + img_path[j][1:]
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
    incl = np.arctan2(np.subtract(y[1:], y[:-1]), np.subtract(x[1:], x[:-1])) * 180 / np.pi
    return np.append(incl, incl[-1])


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


def pre_process(imgs, sets, keys):
    """
    Gaussian blur, edge detection and image resize
    :param imgs:
    :param sets:

    :return:
    """
    if keys.get('shape'):
        shape = sets[keys['shape']]
        imgs = [cv.resize(img, shape) for img in imgs]
    if keys.get('blur'):
        imgs = [cv.GaussianBlur(img, (5, 5), 0) for img in imgs]
    if keys.get('edge_range'):
        lims = sets[keys['edge_range']]
        imgs = [cv.Canny(img, lims[0], lims[1]) for img in imgs]

    return imgs


def image_split(image, overlap=None, blind=0):
    '''
    Splits an image to 2 left and right part evenly when no overlap is provided.
    :param image: Image to split. 2 dimentional ndarray
    :param overlap: Degrees of overlap between the 2 images
    :param blind: Degrees of blind visual field at the back of the agent
    :return:
    '''
    num_of_cols = image.shape[1]
    if blind:
        num_of_cols_perdegree = int(num_of_cols / 360)
        blind_pixels = blind * num_of_cols_perdegree
        blind_pixels_per_side = int(blind_pixels/2)
        image = image[:, blind_pixels_per_side:-blind_pixels_per_side]

    num_of_cols = image.shape[1]
    split_point = int(num_of_cols / 2)
    if overlap:
        num_of_cols_perdegree = int(num_of_cols / (360-blind))
        pixel_overlap = overlap * num_of_cols_perdegree
        l_split = split_point + int(pixel_overlap/2)
        r_split = split_point - int(pixel_overlap/2)
        left = image[:, :l_split]
        right = image[:, r_split:]
    else:
        left = image[:, :split_point]
        right = image[:, split_point:]

    return left, right


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
        num_of_cols_perdegree = num_of_cols / 360
        cols_to_shift = num_of_cols - round(d * num_of_cols_perdegree)
        img1 = image[:, cols_to_shift:num_of_cols]
        img2 = image[:, 0: cols_to_shift]
        return np.concatenate((img1, img2), axis=1)
    else:
        num_of_cols = image.shape[1]
        num_of_cols_perdegree = num_of_cols / 360
        cols_to_shift = round(d * num_of_cols_perdegree)
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


def idf2(img, ref_img):
    """
    Image Differencing Function AMSE
    :param img:
    :param ref_img:
    :return:
    """
    return abs(ref_img - img).mean()


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

def r_cor_coef(ref_img, current_img,  degrees, step):
    '''
    Calculates rotational correlation coefficients
    :param ref_img:
    :param current_img:
    :param degrees:
    :param step:
    :return:
    '''
    degrees = round(degrees/2)  # degrees to rotate for left and right
    r_coef = []   # Hold the r_coefs between the current and the image of the route for every degree
    for k in range(-degrees, degrees, step):
        curr_image = rotate(k, current_img)    #Rotate the current image
        # coe_coef function to find the correlation between the selected route image and the rotated current
        r_coef.append(cor_coef(curr_image, ref_img))
    return r_coef


def ridf(ref_img, current_img,  degrees, step):
    degrees = round(degrees/2)
    rmse = []   # Hold the RMSEs between the current and the image of the route for every degree
    for k in range(-degrees, degrees, step):
        curr_image = rotate(k, current_img)    #Rotate the current image
        rmse.append(idf2(curr_image, ref_img))  #IDF function to find the error between the selected route image and the rotated current
        #TODO: Need to include options for using multiple idf functions.
    return rmse


def flatten_imgs(imgs):
    return [img.flatten() for img in imgs]


def cross_corr(sub_series, series):
    return [np.dot(s, sub_series) for s in series]


def degree_error_logs(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings, degree_thresh=30):
    k = []  # Holds the position of the memory with the shortest distance to the wg position
    logs = {'x_route': [], 'y_route': [], 'route_heading': [], 'route_idx': [],
            'x_grid': [], 'y_grid': [], 'grid_heading': [], 'grid_idx': [], 'errors': []}
    route_end = len(x_route_cords)
    search_step = 10
    memory_pointer = 0
    limit = memory_pointer + search_step
    for i in range(0, len(x_cords)):  # For every grid position
        distance = []
        for j in range(memory_pointer, limit):  # For every route position
            d = math.sqrt((x_cords[i] - x_route_cords[j]) ** 2 + ((y_cords[i] - y_route_cords[j]) ** 2))
            distance.append(d)
        k.append(distance.index(min(distance)) + memory_pointer)
        memory_pointer = k[-1]
        limit = memory_pointer + search_step
        if limit > route_end: limit = route_end
        error = (180 - abs(abs(recovered_headings[i] - route_heading[k[-1]]) - 180))
        if error > degree_thresh or error < -degree_thresh:
            logs['x_route'].append(x_route_cords[k[-1]])
            logs['y_route'].append(y_route_cords[k[-1]])
            logs['route_heading'].append(route_heading[k[-1]])
            logs['route_idx'].append(k[-1])
            logs['x_grid'].append(x_cords[i])
            logs['y_grid'].append(y_cords[i])
            logs['grid_heading'].append(recovered_headings[i])
            logs['grid_idx'].append(i)
            logs['errors'].append(error)
    return logs


def degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings):
    k = []  # Holds the index of the memory with the shortest distance to the grid position
    errors = []  # Holds the error between the world grid image and the closest route image
    route_end = len(x_route_cords)
    search_step = 10
    memory_pointer = 0
    limit = memory_pointer + search_step
    # TODO: Need to iterate the total number of window comparisons instead of the total coord points
    #   they are not equal for current window seq. algorithms
    for i in range(0, len(x_cords)):  # For every grid position
        distance = []
        for j in range(memory_pointer, limit):  # For every route position
            d = math.sqrt((x_cords[i] - x_route_cords[j]) ** 2 + ((y_cords[i] - y_route_cords[j]) ** 2))
            distance.append(d)
        k.append(distance.index(min(distance)) + memory_pointer)
        errors.append(180 - abs(abs(recovered_headings[i] - route_heading[k[-1]]) - 180))
        memory_pointer = k[-1]
        limit = memory_pointer + search_step
        if limit > route_end: limit = route_end
    return errors, k


def mean_degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings):
    error = degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings)
    return sum(error) / len(error)


def check_for_dir_and_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_loop_route(route_dir, route_id=1, grid_pos_limit=100):
    # Path/ Directory settings
    route_id_dir = 'route_' + str(route_id) + '/'
    csv_file = 'route_' + str(route_id) + '.csv'
    route_dir = route_dir + route_id_dir
    grid_dir = '/home/efkag/PycharmProjects/ant_world_alg_bench/AntWorld/world5000_grid/'

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
    X = data[:, 0]  # x location of the image in the world_grid
    Y = data[:, 1]  # y location of the image in the world_grid
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
        temp = route_dir + imgs_route_path[i]
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
    for i in range(0, len(X_route), 1):
        for j in range(0, len(X), 1):
            d = (math.sqrt((X_route[i] - X[j]) ** 2 + (Y_route[i] - Y[j]) ** 2))
            if 1 < d < grid_pos_limit:  # Maximum distance limit from the Route images
                X_inlimit.append(X[j])
                Y_inlimit.append(Y[j])
                X[j] = 0
                Y[j] = 0
                img_dir = grid_dir + img_path[j][1:]
                img = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
                # Normalize
                # img = img * max_norm / img.max()
                world_grid_imgs.append(img)

    return world, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, Heading_route, route_images