import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


w = 20
mu = 0
sig = 1
rv = norm(loc=mu, scale=sig)

ws = [15, 20, 25, 30]
for w in ws:
    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), w)
    pdf = rv.pdf(x)
    pdf = 1 - pdf
    plt.plot(range(w), pdf)
    plt.scatter(range(w), pdf)


plt.show()