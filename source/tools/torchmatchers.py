import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_start_method('spawn')
import numpy as np


def rotate(d, image):
    """
    Converts the degrees into columns and rotates the image.
    Positive degrees rotate the image clockwise
    and negative degrees rotate the image counter clockwise
    :param d: number of degrees the agent will rotate its view
    :param image: An np.array that we want to shift.
    :return: Returns the rotated image.
    """
    num_of_cols = image.shape[1]
    num_of_cols_perdegree = num_of_cols / 360
    cols_to_shift = int(round(d * num_of_cols_perdegree))
    return torch.roll(image, -cols_to_shift, dims=1)


def mae(a, b):
    """
    Image Differencing Function MAE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    return torch.mean(torch.abs(torch.sub(a, b)), axis=(1, 2)).detach().cpu().numpy()


def dot_dist(a, b):
    """
    Returns the dot product distance.
    This function assumes the vectors have zero means and unit variance.
    :param a: numpy vector or matrix
    :param b: numpy vector or matrix
    :return: distance between [0, 2]
    """
    #a = torch.unsqueeze(a, 0)
    a = a.flatten()
    b = b.flatten(start_dim=1)
    res = 1 - torch.matmul(b, a)
    return res.detach().cpu().numpy()

def pick_im_matcher(im_matcher=None):
    matchers = {'dot': dot_dist, 'mae': mae}
    if not matchers.get(im_matcher):
        raise Exception('Non valid matcher method name')
    return matchers.get(im_matcher)

def rmf(query_img, ref_imgs, matcher=mae, d_range=(-180, 180), d_step=1):
    """
    Rotational Matching Function.
    Rotates a query image and compares it with one or more reference images
    :param query_img:
    :param ref_imgs:
    :param matcher:
    :param d_range:
    :param d_step:
    :return:
    """
    query_img = torch.Tensor(query_img)
    query_img = query_img.cuda()
    if ref_imgs.ndim < 3:
      torch.unsqueeze(ref_imgs, 0)

    degrees = range(*d_range, d_step)
    total_search_angle = round((d_range[1] - d_range[0]) / d_step)
    ridfs = np.empty((len(ref_imgs), total_search_angle), dtype=np.float32)

    for i, rot in enumerate(degrees):
        # rotated query image
        rqimg = rotate(rot, query_img)
        ridfs[:, i] = matcher(rqimg, ref_imgs)

    return ridfs if ridfs.shape[0] > 1 else ridfs[0]

def is_cuda_avail():
    return torch.cuda.is_available()