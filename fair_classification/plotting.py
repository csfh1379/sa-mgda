import numpy as np
from matplotlib import pyplot as plt
import os


home_dir = os.path.expanduser('~')
data = []
load_data_dir = os.path.join(home_dir, 'results/fashion')
file_list = ['MGDA_-1.000_8000_0.010.npz', 'MGDA_1.000_8000_0.010.npz', 'MGDA_0.100_8000_0.010.npz',
             'MGDA_0.010_8000_0.010.npz', 'MGDA_0.000_8000_0.010.npz', 'SA-MGDA_8000_0.010.npz']

# file_list = ['MGDA_-1.000_40000_0.001.npz', 'MGDA_1.000_40000_0.001.npz', 'MGDA_0.100_40000_0.001.npz',
#              'MGDA_0.010_40000_0.001.npz', 'MGDA_0.000_40000_0.001.npz', 'SA-MGDA_40000_0.001.npz']

for i in range(len(file_list)):
    data.append(np.load(os.path.join(load_data_dir, file_list[i])))

accAll = []
mean = []
std = []

for i in range(len(data)):
    accAll.append(data[i]['accAll'][:, :, [0, 4, 6]])
for i in range(len(data)):
    min_val = np.min(accAll[i], 2)
    mean.append(np.mean(min_val, 0))
    std.append(np.std(min_val, 0))

fig, ax = plt.subplots()

num_iter = [100 * x for x in range(1, 81)]
# num_iter = [100 * x for x in range(1, 401)]

color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:purple']
label = [r'GD', r'MGDA, $\lambda=1$', r'MGDA, $\lambda=0.1$', r'MGDA, $\lambda=0.01$', r'MGDA, $\lambda=0$', 'SA-MGDA']
for i in range(len(data)):
    ax.plot(num_iter, mean[i], color=color[i], label=label[i])
    ax.fill_between(num_iter, mean[i] + std[i], mean[i] - std[i], facecolor=color[i], alpha=0.15)

ax.set_title('The number of correctly classified data for the worst category')
ax.set_xlabel('Iterations')
ax.set_ylabel('Number of data')
ax.set_xlim(0, 8000)
# ax.set_xlim(0, 40000)
ax.set_ylim(600, 800)
ax.legend(loc='lower right')
plt.show()
