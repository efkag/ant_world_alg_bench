import numpy as np


def get_ridf_depths(ridfs: np.ndarray):
    ridfs = np.array(ridfs)
    if len(ridfs.shape) < 2:
        ridfs = np.expand_dims(ridfs, axis=0)
    ridfs = ridfs - np.expand_dims(np.min(ridfs, axis=1), axis=1)
    depths = np.max(ridfs, axis=1)-np.min(ridfs, axis=1)
    return depths