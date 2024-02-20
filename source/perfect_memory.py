from source.utils import pick_im_matcher, mae, rmse, cor_dist, rmf, nanmae
import numpy as np
from source.navs.navs import Navigator
from source.utils import dot_dist
from source.imgproc import Pipeline


class PerfectMemory(Navigator):

<<<<<<< HEAD
    def __init__(self, route_images, **kwargs):
        super().__init__(route_images, **kwargs)
=======
class PerfectMemory:

    def __init__(self, route_images, matching, deg_range=(-180, 180), deg_step=1, **kwargs):
        self.route_images = route_images
        self.deg_step = deg_step
        self.deg_range = deg_range
        self.degrees = np.arange(*deg_range)
>>>>>>> Update nav serach range
        self.recovered_heading = []
        self.logs = []
        self.matched_index_log = []
        self.argminmax = np.argmin
        self.best_sims = []
        # if the dot product distance is used we need to make sure the images are standardized
        if self.matcher == dot_dist:
            pipe = Pipeline(normstd=True)
            self.route_images = pipe.apply(route_images)

    def get_heading(self, query_img):
        # get the rotational similarities between a query image and a window of route images
        rsims = rmf(query_img, self.route_images, self.matcher, self.deg_range, self.deg_step)

        # Holds the best rot. match between the query image and route images
        mem_sims = []
        # Recovered headings for the current image
        mem_headings = []
        # get best similarity match adn index w.r.t degrees
        indices = self.argminmax(rsims, axis=1)
        for i, idx in enumerate(indices):
            mem_sims.append(rsims[i, idx])
            mem_headings.append(self.degrees[idx])

        # append the rsims of all window route images for that query image
        self.logs.append(rsims)
        # find best image match and heading
        index = int(self.argminmax(mem_sims))
        self.best_sims.append(mem_sims[index])
        heading = mem_headings[index]
        self.recovered_heading.append(heading)
        # Update memory pointer
        self.matched_index_log.append(index)
        return heading
    
    def navigate(self, query_imgs):
        assert isinstance(query_imgs, list)
        for query_img in query_imgs:
            self.get_heading(query_img)
        return self.recovered_heading

    # def navigate(self, query_imgs):
    # #TODO: need to be updated to use get heading in a loop
    #     assert isinstance(query_imgs, list)

    #     # For every query image image
    #     for query_img in query_imgs:

    #         # get the rotational similarities between a query image and a list of all route images
    #         rsims = rmf(query_img, self.route_images, self.matcher, self.deg_range, self.deg_step)

    #         # Holds the best rot. similarity between the query image and route images
    #         mem_sims = []
    #         # Hold the recovered Headings for the current image by the different route images
    #         mem_headings = []

    #         indices = self.argminmax(rsims, axis=1)
    #         for i, idx in enumerate(indices):
    #             mem_sims.append(rsims[i, idx])
    #             mem_headings.append(self.degrees[idx])

    #         # append the rsims between one query image and all route images
    #         self.logs.append(rsims)
    #         # Get best image match
    #         index = int(self.argminmax(mem_sims))
    #         # Get best image heading
    #         self.recovered_heading.append(mem_headings[index])
    #         self.matched_index_log.append(index)

    #     return self.recovered_heading

    def get_rec_headings(self): return self.recovered_heading

    def get_index_log(self): return self.matched_index_log

    def get_rsims_log(self): return self.logs

    def get_window_log(self): return None

    def reset_window(self, pointer):
        pass

    def get_best_sims(self):
        return self.best_sims
    
    def get_name(self):
        return 'PM'
