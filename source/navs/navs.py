import numpy as np
from source.utils import dot_dist
from source.imgproc import Pipeline
from abc import ABC, abstractmethod
from source.utils import pick_im_matcher
from source.utils import rmf
from source.tools import torchmatchers
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Navigator():
    
    def __init__(self, route_images, matcher='mae', deg_range=(-180, 180), degree_shift=1, **kwargs) -> None:
        self.route_end = len(route_images)
        self.route_images = route_images
        self.deg_range = deg_range
        self.deg_step = degree_shift
        self.degrees = np.arange(*deg_range)
        self.matcher = pick_im_matcher(matcher)
        # if the dot product distance is used we need to make sure the images are standardized
        if self.matcher == dot_dist:
            self.pipe = Pipeline(normstd=True)
            self.route_images = self.pipe.apply(route_images)
        else: 
            self.pipe = Pipeline()
        self.argminmax = np.argmin
        self.rmf = rmf
        self.using_torch = False

        if torch.cuda.is_available():
            self.using_torch = True
            self.route_images = np.array(self.route_images)
            self.route_images = torch.Tensor(self.route_images)
            self.route_images = self.route_images.cuda()
            self.matcher = torchmatchers.pick_im_matcher(matcher)
            self.rmf = torchmatchers.rmf
    
    @abstractmethod
    def get_heading():
        pass

    @abstractmethod
    def navigate():
        pass

    @abstractmethod
    def get_rec_headings(self):
        pass
    
    @abstractmethod
    def get_index_log(self):
        pass

    @abstractmethod
    def get_rsims_log(self):
        pass
    
    @abstractmethod
    def get_best_sims(self):
        pass

    @abstractmethod
    def get_time_com(self):
        pass
    
    @abstractmethod
    def get_name(self):
        pass

    def clear_route_mem(self):
        del self.route_imgs
        if self.using_torch:
            torch.cuda.empty_cache()

