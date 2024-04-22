import torch


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
    return torch.mean(torch.abs(a - b), dim=-1)