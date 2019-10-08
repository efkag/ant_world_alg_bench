import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd



def display_image(image, size=(10, 10)):
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
    plt.savefig("test.png")
    plt.show()


def plot_map(world, route_cords=None, grid_cords=None, size=(15, 15), save=False, zoom=(), zoom_factor=1000,
             vectors=None, grid_vectors=None, marker_size=10, scale=40):
    fig = plt.figure(figsize=size)
    fig.suptitle('World Grid', fontsize=16, fontweight='bold')
    plt.xlabel('x coordinates', fontsize=14, fontweight='bold')
    plt.ylabel('y coordinates', fontsize=13, fontweight='bold')
    # Plot circles for route image locations
    if route_cords and not vectors: plt.scatter(route_cords[0], route_cords[1], marker="o", s=marker_size, color='blue')
    # Plot stars for grid image locations
    if grid_cords and not grid_vectors: plt.scatter(grid_cords[0], grid_cords[1], marker="*", s=marker_size,
                                                    color='red')
    # Plot route images heading vectors
    if vectors: plt.quiver(route_cords[0], route_cords[1], vectors[0], vectors[1], scale=scale, color='b')
    # Plot world grid images heading vectors
    if grid_vectors: plt.quiver(grid_cords[0], grid_cords[1], grid_vectors[0], grid_vectors[1], scale=scale, color='r')

    plt.imshow(world, zorder=0, extent=[-0.158586 * 1000, 10.2428 * 1000, -0.227704 * 1000, 10.0896 * 1000])
    if save: fig.savefig('Graph1.jpg')
    if zoom:
        plt.xlim([zoom[0] - zoom_factor, zoom[0] + zoom_factor])
        plt.ylim([zoom[1] - zoom_factor, zoom[1] + zoom_factor])
    plt.savefig("test.png")
    plt.show()


def load_route(route_id, grid_pos_limit=200):
    # Path/ Directory settings
    route_id_dir = 'ant1_route' + route_id + '/'
    route_dir = 'Datasets/AntWorld/' + route_id_dir
    grid_dir = 'Datasets/AntWorld/world5000_grid/'

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
        img = cv2.imread(route_dir + imgs_route_path[i][1:], cv2.IMREAD_GRAYSCALE)
        # Normalize
        img = img * max_norm / img.max()
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
        if (min(dist) < grid_pos_limit):  # Maximum distance limit from the Route images
            X_inlimit.append(X[i])
            Y_inlimit.append(Y[i])
            img_dir = grid_dir + img_path[i][1:]
            img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            # Normalize
            img = img * max_norm / img.max()
            world_grid_imgs.append(img)

    return world, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, Heading_route, route_images