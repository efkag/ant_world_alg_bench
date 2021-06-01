import numpy as np
import cv2 as cv
import pandas as pd
from scipy.spatial.distance import cdist
from source.utils import calc_dists, travel_dist, pre_process, angular_error, seq_angular_error, travel_dist
import os

class Route:
    def __init__(self, path, route_id, read_imgs=True, grid_path=None, max_dist=0.2):
        self.path = path
        self.grid_path = grid_path
        self.read_imgs = read_imgs
        self.max_dist = max_dist
        self.proc_imgs = []
        self.proc_qimgs= []
        self.route_id = str(route_id)
        self.route_dict = self.load_route()
        self.points_on_route = len(self.route_dict['x'])
        self.is_segmented = False
        self.segment_indices = None
        self.no_of_segments = None

    def load_route(self):
        route_data = pd.read_csv(self.path + 'route' + self.route_id + '.csv', index_col=False)
        route_data = route_data.to_dict('list')
        # convert the lists to numpy arrays
        for k in route_data:
            route_data[k] = np.array(route_data[k])
        if self.read_imgs:
            imgs = []
            for i in route_data['filename']:
                img = cv.imread(self.path + i, cv.IMREAD_GRAYSCALE)
                imgs.append(img)
            route_data['imgs'] = imgs

        # Sample positions and images from the grid near the route for testing
        if self.grid_path:
            grid = pd.read_csv(self.grid_path + '/grid.csv')
            grid = grid.to_dict('list')
            for k in grid:
                grid[k] = np.array(grid[k])

            grid_xy = np.transpose(np.array([grid['x'], grid['y']]))
            query_indexes = np.empty(0, dtype=int)
            qx = []
            qy = []
            qimg = []
            # Fetch images from the grid that are located nearby route images.
            # for each route position
            for i, (x, y) in enumerate(zip(route_data['x'], route_data['y'])):
                # get distance between route point and all grid points
                dist = np.squeeze(cdist([(x, y)], grid_xy, 'euclidean'))
                # indexes of distances within the limit
                indexes = np.where(dist <= self.max_dist)[0]
                # check which indexes have not been encountered before
                mask = np.isin(indexes, query_indexes, invert=True)
                # get the un-encountered indexes
                indexes = indexes[mask]
                # save the indexes
                query_indexes = np.append(query_indexes, indexes)

                for i in indexes:
                    qx.append(grid_xy[i, 0])
                    qy.append(grid_xy[i, 1])
                    imgfile = os.path.join(self.grid_path, grid['filename'][i])
                    qimg.append(cv.imread(imgfile, cv.IMREAD_GRAYSCALE))

            route_data['qx'] = np.array(qx)
            route_data['qy'] = np.array(qy)
            route_data['qimgs'] = qimg

        return route_data

    def imgs_pre_proc(self, params):
        if self.route_dict.get('qimgs'):
            self.proc_qimgs = pre_process(self.route_dict['qimgs'], params)
        self.proc_imgs = pre_process(self.route_dict['imgs'], params)
        return self.proc_imgs

    def segment_route(self, segment_size_m):
        self.is_segmented = True
        dist = travel_dist(self.route_dict['x'], self.route_dict['y'])
        self.no_of_segments = int(round(dist / segment_size_m))

        # xs = np.array_split(self.route_dict['x'], no_of_segments)
        # ys = np.array_split(self.route_dict['y'], no_of_segments)
        # hs = np.array_split(self.route_dict['yaw'], no_of_segments)
        # subroute = {}

        # segment size in indices
        index_segment_size = int(self.points_on_route / self.no_of_segments)

        # get starting indices and coords for each segment
        indices = [0]
        staring_coords = []
        coord_dict = {'x': self.route_dict['x'][0],
                      'y': self.route_dict['y'][0],
                      'yaw': self.route_dict['yaw'][0]}
        staring_coords.append(coord_dict)
        for i in range(1, self.no_of_segments-1):
            indices.append(indices[-1] + index_segment_size)
            coord_dict = {'x':self.route_dict['x'][indices[-1]],
                          'y':self.route_dict['y'][indices[-1]],
                          'yaw':self.route_dict['yaw'][indices[-1]]}
            staring_coords.append(coord_dict)
        self.segment_indices = indices
        return indices, staring_coords

    def calc_errors(self, trajectory):
        if self.is_segmented:
            return angular_error(self.route_dict, trajectory)
        else:
            return seq_angular_error(self.route_dict, trajectory)

    def get_tavel_distance(self):
        return travel_dist(self.route_dict['x'], self.route_dict['y'])

    def get_route_dict(self):
        return self.route_dict

    def get_imgs(self):
        return self.route_dict['imgs']

    def get_qimgs(self):
        return self.route_dict['qimgs']

    def get_qxycoords(self):
        return {'x': self.route_dict['qx'], 'y': self.route_dict['qy']}

    def get_xycoords(self):
        return {'x': self.route_dict['x'], 'y': self.route_dict['y']}

