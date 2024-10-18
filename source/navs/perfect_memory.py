from .navs import Navigator
import time
import numpy as np
from source.tools.metrics import get_ridf_depths

class PerfectMemory(Navigator):

    def __init__(self, route_images, **kwargs):
        super().__init__(route_images, **kwargs)
        self.recovered_heading = []
        self.logs = []
        self.matched_index_log = []
        self.best_sims = []
        self.best_ridfs = []
        self.time_com = []

    def get_heading(self, query_img):
        start_time = time.perf_counter()
        
        query_img = self.pipe.apply(query_img)
        # get the rotational similarities between a query image and a window of route images
        rsims = self.rmf(query_img, self.route_images, self.matcher, self.deg_range, self.deg_step)

        # get best similarity match adn index w.r.t degrees
        indices = self.argminmax(rsims, axis=1)
        mem_sims = rsims[np.arange(0, self.route_end), indices]
        mem_headings = self.degrees[indices]

        # append the rsims of all window route images for that query image
        #self.logs.append(rsims)
        # find best image match and heading
        depths = get_ridf_depths(rsims)
        idx = np.argmax(depths)

        self.best_ridfs.append(rsims[idx])
        self.best_sims.append(mem_sims[idx])
        heading = mem_headings[idx]
        self.recovered_heading.append(heading)
        # Update memory pointer
        self.matched_index_log.append(idx)
        end_time = time.perf_counter()
        self.time_com.append((end_time-start_time))
        return heading
    
    def navigate(self, query_imgs):
        assert isinstance(query_imgs, list)
        for query_img in query_imgs:
            self.get_heading(query_img)
        return self.recovered_heading

    def get_rec_headings(self): return self.recovered_heading

    def get_index_log(self): return self.matched_index_log

    def get_rsims_log(self): return self.logs

    def get_window_log(self): return []

    def reset_window(self, pointer):
        pass

    def get_best_sims(self):
        return self.best_sims
    
    def get_best_ridfs(self):return self.best_ridfs

    def get_time_com(self):
        return self.time_com
    
    def get_name(self):
        return 'PM'
