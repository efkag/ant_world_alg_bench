from source.utils import mae, rmse, cor_dist, rmf
import numpy as np



class PerfectMemory:

    def __init__(self, route_images, matching, deg_range=(0, 360), deg_step=1):
        self.route_images = route_images
        self.deg_step = deg_step
        self.deg_range = deg_range
        self.degrees = np.arange(*deg_range)
        self.recovered_heading = []
        self.logs = []
        self.matched_index_log = []
        matchers = {'corr': cor_dist, 'rmse': rmse, 'mae':mae}
        self.matcher = matchers.get(matching)
        if not self.matcher:
            raise Exception('Non valid matcher method name')
        self.argminmax = np.argmin

    def navigate(self, query_imgs):

        assert isinstance(query_imgs, list)

        # For every query image image
        for query_img in query_imgs:

            # get the rotational similarities between a query image and a list of all route images
            rsims = rmf(query_img, self.route_images, self.matcher, self.deg_range, self.deg_step)

            # Holds the best rot. similarity between the query image and route images
            mem_sims = []
            # Hold the recovered Headings for the current image by the different route images
            mem_headings = []

            for rsim in rsims:
                index = int(self.argminmax(rsim))
                mem_sims.append(rsim[index])
                mem_headings.append(self.degrees[index])

            # append the rsims between one query image and all route images
            self.logs.append(rsims)
            # Get best image match
            index = int(self.argminmax(mem_sims))
            # Get best image heading
            self.recovered_heading.append(mem_headings[index])
            self.matched_index_log.append(index)

        return self.recovered_heading, self.logs

    def get_index_log(self): return self.matched_index_log
