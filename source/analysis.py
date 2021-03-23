import numpy as np


def iqr_outliers(data):
    q3 = np.percentile(data, 75)
    q1 = np.percentile(data, 25)
    iqr = q3 - q1
    outliers_over = data[data > (q3 + 1.5 * iqr)]
    outliers_under = data[data < (q1 - 1.5 * iqr)]
    out = np.append(outliers_over, outliers_under)
    return out


def perc_outliers(data):
    out = iqr_outliers(data)
    perc = len(out)/len(data)
    return perc

