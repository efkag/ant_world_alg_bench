from source.utils import pick_im_matcher, mae, rmse, cor_dist, rmf, nanmae
import numpy as np
from source.navs.navs import Navigator
from source.utils import dot_dist
from source.imgproc import Pipeline


class PerfectMemory(Navigator):

    def __init__(self, route_images, **kwargs):
        super().__init__(route_images, **kwargs)
        self.recovered_heading = []
        self.logs = []
        self.matched_index_log = []
        self.argminmax = np.argmin
        self.best_sims = []
        # if the dot product distance is used we need to make sure the images are standardized
        if self.matcher == dot_dist:
            self.pipe = Pipeline(normstd=True)
            self.route_images = self.pipe.apply(route_images)
        else: 
            self.pipe = Pipeline()

    def get_heading(self, query_img):
        query_img = self.pipe.apply(query_img)
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
