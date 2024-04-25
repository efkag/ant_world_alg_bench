import numpy as np
from abc import ABC, abstractmethod
from source.utils import pick_im_matcher
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Navigator():
    
    def __init__(self, route_images, matcher='mae', deg_range=(-180, 180), degree_shift=1, **kwargs) -> None:
        self.route_end = len(route_images)
        self.route_images = route_images
        self.deg_range = deg_range
        self.deg_step = degree_shift
        self.degrees = np.arange(*deg_range)
        self.matcher = pick_im_matcher(matcher)
        self.argminmax = np.argmin
        self.using_torch = False

        if torch.cuda.is_available():
            self.using_torch = True
            import pdb;  pdb.set_trace()
            self.route_images = np.array(self.route_images)
            self.route_images = torch.Tensor(self.route_images)
    
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
    def get_name(self):
        pass

