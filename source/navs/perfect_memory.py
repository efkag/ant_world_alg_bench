from .navs import Navigator
import time
import numpy as np
from source.tools.metrics import get_ridf_depths

class PerfectMemory(Navigator):

    def __init__(self, route_images, match_type='ridf_depth', **kwargs):
        super().__init__(route_images, **kwargs)
        self.recovered_heading = []
        self.logs = []
        self.matched_index_log = []
        self.best_sims = []
        self.best_ridfs = []
        self.time_com = []

        match_methods = {'ridf_min':self.window_match_minima, 
                       'ridf_depth':self.window_match_depth}
        if match_type not in match_methods.keys():
            raise Exception('Non valid window match method type')
        self.get_window_match = match_methods.get(match_type)

    def get_heading(self, query_img):
        start_time = time.perf_counter()
        
        query_img = self.pipe.apply(query_img)
        # get the rotational similarities between a query image and a window of route images
        ridfs = self.rmf(query_img, self.route_images, self.matcher, self.deg_range, self.deg_step)

        # get best similarity match adn index w.r.t degrees
        indices = self.argminmax(ridfs, axis=1)
        mem_sims = ridfs[np.arange(0, self.route_end), indices]
        mem_headings = self.degrees[indices]

        # append the ridfs of all window route images for that query image
        #self.logs.append(ridfs)
        # find best image match index
        idx = self.get_window_match(ridfs)

        self.best_ridfs.append(ridfs[idx])
        self.best_sims.append(mem_sims[idx])
        heading = mem_headings[idx]
        self.recovered_heading.append(heading)
        # Update memory pointer
        self.matched_index_log.append(idx)
        end_time = time.perf_counter()
        self.time_com.append((end_time-start_time))
        return heading
    
    def window_match_depth(self, wridfs):
        depths = get_ridf_depths(wridfs)
        return np.argmax(depths)
    
    def window_match_minima(self, wridfs):
        idx = np.unravel_index(np.argmin(wridfs, axis=None), wridfs.shape)
        return idx[0]
    
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
