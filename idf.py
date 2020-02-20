from math import sqrt


class RotationalIDF:

    def match(self, img, ref_img):
        """
        Image Differencing Function RMSE
        :param img:
        :param ref_img:
        :return:
        """
        return sqrt(((ref_img - img) ** 2).mean())

    @staticmethod
    def best_match(array):
        minimum = min(array)
        return minimum, array.index(minimum)
