import numpy as np
from matplotlib import pyplot as plt


w = [10]
new_w = w[0]
for i in range(200):
    #new_w = round(w[-1] - (np.log(w[-1])))
    new_w = round(w[-1] + 10/np.log(w[-1]))
    #new_w += round(10/np.log(w[-1]))
    w.append(new_w)

print(w)

plt.scatter(range(len(w)), w)
plt.plot(w)
plt.show()

plt.plot(np.diff(w))
plt.show()