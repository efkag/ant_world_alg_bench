import numpy as np


class CorrelationCoefficient:

    def cov(self, a, b):
        """
        Calculates covariance (non sample)
        Assumes flattened arrays
        :param a:
        :param b:
        :return:
        """
        if len(a) != len(b):
            return

        a_mean = np.mean(a)
        b_mean = np.mean(b)

        return np.sum((a - a_mean) * (b - b_mean)) / (len(a))

    def match(self, a, b):
        """
        Calculate correlation coefficient
        :param a:
        :param b:
        :return:
        """
        a = a.flatten()
        b = b.flatten()
        return self.cov(a, b) / (np.std(a) * np.std(b))


