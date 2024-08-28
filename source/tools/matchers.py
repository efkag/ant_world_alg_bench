import numpy as np
from numpy.linalg import norm
from scipy.stats import circmean


def rotate(d, image):
    """
    Converts the degrees into columns and rotates the image.
    Positive degrees rotate the image clockwise
    and negative degrees rotate the image counter clockwise
    :param d: number of degrees the agent will rotate its view
    :param image: An np.array that we want to shift.
    :return: Returns the rotated image.
    """
    assert abs(d) <= 360

    num_of_cols = image.shape[1]
    num_of_cols_perdegree = num_of_cols / 360
    cols_to_shift = int(round(d * num_of_cols_perdegree))
    return np.roll(image, -cols_to_shift, axis=1)


def mse(a, b):
    """
    Image Differencing Function MSE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.mean(np.subtract(ref_img, a)**2) for ref_img in b]

    return np.mean(np.subtract(a, b)**2)


def weighted_mse(a, b, weights=None):
    if isinstance(b, list):
        return [np.mean(weights * np.subtract(ref_img, a)**2) for ref_img in b]

    return np.mean(weights * np.subtract(a, b)**2)


def rmse(a, b):
    """
    Image Differencing Function RMSE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.sqrt(np.mean(np.subtract(ref_img, a)**2)) for ref_img in b]

    return np.sqrt(np.mean(np.subtract(a, b)**2))


def mae(a, b):
    """
    Image Differencing Function MAE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.mean(np.abs(a - img)) for img in b]

    return np.mean(np.abs(a - b))


def nanmae(a, b):
    """
    Image Differencing Function MAE for images with nan values
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.nanmean(np.abs(a - img)) for img in b]

    return np.nanmean(np.abs(a - b))


def cov(a, b):
    """
    Calculates covariance (non sample)
    Assumes flattened arrays
    :param a:
    :param b:
    :return:
    """
    assert len(a) == len(b)

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    return np.sum((a - a_mean) * (b - b_mean)) / (len(a))


def cor_coef(a, b):
    """
    Calculate the correlation coefficient
    :param a: A single image or vector
    :param b: A single image or vector
    :return:
    """
    a = a.flatten()
    b = b.flatten()
    return cov(a, b) / (np.std(a) * np.std(b))


def cor_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between a (list of) vector(s) b and reference vector a
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [_cc_dist(a, img) for img in b]

    return _cc_dist(a, b)

def _cc_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between two vectors.
    :param a:
    :param b:
    :return:
    """
    amu = np.mean(a)
    bmu = np.mean(b)
    a = a - amu
    b = b - bmu
    ab = np.mean(a * b)
    avar = np.mean(np.square(a))
    bvar = np.mean(np.square(b))
    return 1.0 - ab / np.sqrt(avar * bvar)


def nan_correlation_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between two vectors.
    Where a and b can contain nan values.
    :param a:
    :param b:
    :return:
    """
    amu = np.nanmean(a)
    bmu = np.nanmean(b)
    a = a - amu
    b = b - bmu
    ab = np.nanmean(a * b)
    avar = np.nanmean(np.square(a))
    bvar = np.nanmean(np.square(b))
    dist = 1.0 - ab / np.sqrt(avar * bvar)
    return dist


def nan_cor_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between a (list of) vector(s) b and reference vector a.
    Where a and b can contain nan values.
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [nan_correlation_dist(a, img) for img in b]

    return nan_correlation_dist(a, b)


# def cos_dist(a, b):
#     """
#     Calculates cosine similarity
#     between a (list of) vector(s) b and reference vector a
#     :param a:
#     :param b:
#     :return:
#     """
#     if isinstance(b, list):
#         return [cosine(a, img) for img in b]

#     return cosine(a, b)

def cos_dist(a, b):
    """
    Cossine Distance
    :param a:
    :param b:
    :return:
    """
    if isinstance(b, list):
        return [1.0 - (np.vdot(a, img) / (norm(a) * norm(img))) for img in b]
    
    return 1.0 - (np.vdot(a, b) / (norm(a) * norm(b)))


def cos_sim(a, b):
    """
    Cossine similarity.
    :param a:
    :param b:
    :return:
    """
    return np.vdot(a, b) / (norm(a) * norm(b))


def dot_dist(a, b):
    """
    Returns the dot product distance.
    This function assumes the vectors have zero means and unit variance.
    :param a: numpy vector or matrix 
    :param b: numpy vector or matrix 
    :return: distance between [0, 2]
    """
    if isinstance(b, list):
        return [1 - np.vdot(a, img) for img in b]
    
    return 1 - np.vdot(a, b)


def entropy_im(img, bins=256):
    #get the histogram
    amarg = np.histogramdd(np.ravel(img), bins = bins)[0]/img.size
    amarg = amarg[np.ravel(amarg) > 0]
    return -np.sum(np.multiply(amarg, np.log2(amarg)))


def mutual_inf(a, b, bins=256):
    if isinstance(b, list):
        return [_mut_inf(a, img, bins) for img in b]

    return mutual_inf(a, b, bins)


def _mut_inf(a, b, bins=256):
    hist_2d, x_edges, y_edges = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    pab = hist_2d / float(np.sum(hist_2d))
    pa = np.sum(pab, axis=1) # marginal for x over y
    pb = np.sum(pab, axis=0) # marginal for y over x
    pa_pb = pa[:, None] * pb[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pab > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pab[nzs] * np.log(pab[nzs] / pa_pb[nzs]))


def entropy_dist(a, b, bins=256):
    if isinstance(b, list):
        return [_entropy_dist(a, img, bins) for img in b]

    return _entropy_dist(a, b, bins)


def _entropy_dist(a, b, bins=256):
    ## how many bins? 256 always? 
    # ask andy here for join entropy vs H(a)
    # amarg = np.histogram(np.ravel(a), bins = bins)[0]/a.size
    # amarg = amarg[amarg > 0]
    # aentropy = -np.sum(np.multiply(amarg, np.log2(amarg)))

    hist_2d, x_edges, y_edges = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    pab = hist_2d / float(np.sum(hist_2d))
    pa = np.sum(pab, axis=1) # marginal for a over b
    pb = np.sum(pab, axis=0) # marginal for b over a
    pa_pb = pa[:, None] * pb[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pab > 0 # Only non-zero pab values contribute to the sum

    pab_joint = pab[np.logical_and(pab, pab)]
    ent_pab = -np.sum(pab_joint * np.log(pab_joint))
    #here from practical rasons i could also subtract from the entropy of a
    return ent_pab - (np.sum(pab[nzs] * np.log(pab[nzs] / pa_pb[nzs])))


def pick_im_matcher(im_matcher=None):
    matchers = {'ccd': cor_dist, 'dot': dot_dist, 'rmse': rmse, 'mse':mse, 'mae': mae, 'entropy':entropy_dist}
    if not matchers.get(im_matcher):
        raise Exception('Non valid matcher method name')
    return matchers.get(im_matcher)

def rmf(query_img, ref_imgs, matcher=mae, d_range=(-180, 180), d_step=1, norm_imgs=False):
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
    assert d_step > 0
    assert not isinstance(query_img, list)
    if not isinstance(ref_imgs, list):
        ref_imgs = [ref_imgs]

    if norm_imgs:
        ref_mean = np.mean(ref_imgs)
        ref_imgs = [im - ref_mean for im in ref_imgs]

    degrees = range(*d_range, d_step)
    total_search_angle = round((d_range[1] - d_range[0]) / d_step)
    sims = np.empty((len(ref_imgs), total_search_angle), dtype=np.float32)

    for i, rot in enumerate(degrees):
        # rotated query image
        rqimg = rotate(rot, query_img)
        sims[:, i] = matcher(rqimg, ref_imgs)

    return sims if sims.shape[0] > 1 else sims[0]