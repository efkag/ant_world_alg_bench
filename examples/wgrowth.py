import numpy as np
from matplotlib import pyplot as plt

gamma = 10
c_update = 5

w_log_growth = [15]
w_log_shrink = [400]
w_lin_growth = [15]
w_lin_shrink = [400]
#new_w = w[0]
reps = 100
for i in range(reps):
    # w_log_growth.append(np.ceil(w_log_growth[-1] + gamma/np.log(w_log_growth[-1])))
    # w_log_shrink.append(max(np.ceil(w_log_shrink[-1] - np.log(w_log_shrink[-1])), gamma))
    
    w_log_growth.append(w_log_growth[-1] + round(gamma/np.log(w_log_growth[-1])))
    w_log_shrink.append(max(w_log_shrink[-1] - round(np.log(w_log_shrink[-1])), gamma))
    
    w_lin_growth.append(w_lin_growth[-1] + c_update)
    w_lin_shrink.append(max(w_lin_shrink[-1] - c_update, gamma))
    #new_w += round(10/np.log(w[-1]))
    #w.append(new_w)

x = range(reps+1)
#plt.scatter(x, w_log_growth)
plt.plot(w_log_growth, label='log growth')
plt.scatter(range(len(w_log_growth)), w_log_growth)
plt.plot(w_log_shrink, label='log shrinkage')
plt.scatter(range(len(w_log_shrink)),w_log_shrink)


plt.plot(w_lin_growth, label='linear growth')
plt.scatter(range(len(w_lin_growth)), w_lin_growth)
plt.plot(w_lin_shrink, label='linear shrinkage')
plt.scatter(range(len(w_lin_shrink)), w_lin_shrink)

plt.xlabel('timesteps')
plt.ylabel('window size')
plt.legend()
plt.tight_layout()
plt.show()

# plt.plot(np.diff(w))
# plt.show()