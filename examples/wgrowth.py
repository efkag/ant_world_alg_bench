import numpy as np
import os
fwd = os.path.dirname(__file__)
from matplotlib import pyplot as plt

gamma = 10
c_update = 5
r = 0.1
mem_size = 600.

w_log_growth = [15.]
w_log_shrink = [mem_size]

w_lin_growth = [15.]
w_lin_shrink = [mem_size]

w_exp_growth = [15.]
w_exp_shrink = [mem_size]
#new_w = w[0]
reps = 100
for i in range(reps):
    # w_log_growth.append(np.ceil(w_log_growth[-1] + gamma/np.log(w_log_growth[-1])))
    # w_log_shrink.append(max(np.ceil(w_log_shrink[-1] - np.log(w_log_shrink[-1])), gamma))
    
    w_log_growth.append(min(w_log_growth[-1] + round(mem_size/(w_log_growth[-1])), mem_size))
    w_log_shrink.append(max(w_log_shrink[-1] - round(mem_size/(w_log_growth[-1])), gamma))
    
    w_lin_growth.append(min(w_lin_growth[-1] + c_update, mem_size))
    w_lin_shrink.append(max(w_lin_shrink[-1] - c_update, gamma))

    w_exp_growth.append(min(w_exp_growth[-1] + round(w_exp_growth[-1] * r), mem_size))
    w_exp_shrink.append(max(w_exp_shrink[-1] - round(w_exp_growth[-1] * r), gamma))
    #new_w += round(10/np.log(w[-1]))
    #w.append(new_w)

x = range(reps+1)
#plt.scatter(x, w_log_growth)
fig, ax = fig, ax = plt.subplots(figsize=(7, 4))
plt.plot(w_log_growth, label='log growth')
plt.scatter(range(len(w_log_growth)), w_log_growth, s=10)
plt.plot(w_log_shrink, label='log shrinkage')
plt.scatter(range(len(w_log_shrink)),w_log_shrink, s=10)


plt.plot(w_lin_growth, label='linear growth')
plt.scatter(range(len(w_lin_growth)), w_lin_growth, s=10)
plt.plot(w_lin_shrink, label='linear shrinkage')
plt.scatter(range(len(w_lin_shrink)), w_lin_shrink, s=10)


plt.plot(w_exp_growth, label='exp growth')
plt.scatter(range(len(w_exp_growth)), w_exp_growth, s=10)
plt.plot(w_exp_shrink, label='exp shrinkage')
plt.scatter(range(len(w_exp_shrink)), w_exp_shrink, s=10)

plt.xlabel('timesteps')
plt.ylabel('window size')
plt.legend()
plt.tight_layout()
io_path = os.path.join(fwd, 'figures' , 'w_growth.png')
fig.savefig(io_path)
io_path = os.path.join(fwd, 'figures' , 'w_growth.pdf')
fig.savefig(io_path)
plt.show()

# plt.plot(np.diff(w))
# plt.show()