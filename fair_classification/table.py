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
    accAll.append(data[i]['accAll'][:, -1, [0, 4, 6]])

categories = ['T-shirt/top', 'Coat', 'Shirt']
for i in range(len(data)):
    min_val = np.min(accAll[i], 1)
    print('Number of correctly classified data by %s' % file_list[i])
    print('Worst: Mean {:.2f}, std {:.2f}'.format(np.mean(min_val), np.std(min_val)))
    for j in range(3):
        print('{:s}: Mean {:.2f}, std {:.2f}'.format(categories[j], np.mean(accAll[i], 0)[j], np.std(accAll[i], 0)[j]))
    print('\n')
