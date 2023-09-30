import numpy as np
from matplotlib import pyplot as plt

gamma = 10
c_update = 5
w = [10]
w_log_growth = [10]
w_log_shrink = [800]
w_lin_growth = [10]
w_lin_shrink = [800]
#new_w = w[0]
reps = 150
for i in range(reps):
    w_log_growth.append(np.ceil(w_log_growth[-1] + gamma/np.log(w_log_growth[-1])))
    w_log_shrink.append(np.ceil(w_log_shrink[-1] - np.log(w_log_shrink[-1])))

    w_lin_growth.append(np.ceil(w_lin_growth[-1] + c_update))
    w_lin_shrink.append(np.ceil(w_lin_shrink[-1] - c_update))
    #new_w += round(10/np.log(w[-1]))
    #w.append(new_w)

x = range(reps+1)
#plt.scatter(x, w_log_growth)
plt.plot(w_log_growth, label='log growth')
plt.plot(w_log_shrink, label='log shrinkage')
plt.plot(w_lin_growth, label='linear growth')
plt.plot(w_lin_shrink, label='linear shrinkage')
plt.legend()
plt.tight_layout()
plt.show()

# plt.plot(np.diff(w))
# plt.show()