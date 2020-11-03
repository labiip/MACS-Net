# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:54:37 2020

@author: z1092
"""

import matplotlib.pyplot as plt
import numpy as np
import utilities
import h5py

# Load HD5F file
h5f = h5py.File('./LowRes_13434_overlapping_pairs.h5','r')
xdata = h5f['dataset_1'][...,0]
labels = h5f['dataset_1'][...,1]
h5f.close()

# Clean labels
labels = utilities.cleanLabelNearestNeighbour_alllabels(labels)

# Crop to 88x88 pixels and save processed numpy arrays
labels = utilities.makeXbyY(labels, 88, 88)
np.save('./ydata_88x88_0123_82146_onehot', labels)
xdata = utilities.makeXbyY(xdata, 88, 88).reshape((13434,88,88, 1))
np.save('./xdata_88x88_82146', xdata)


# Padding to 128×128 pixels
a=np.load('./xdata_88x88.npy') 
x=a[0,:,:,0]
y=np.zeros((13434, 128, 128, 1),dtype='int64')
y[:, 20:108, 20:108, :]=a
x1=y[0,:,:,0]
np.save('./xdata_128x128.npy', y)

b=np.load('./xdata_88x88.npy') 
y1=np.zeros((13434, 128, 128, 4),dtype='float64')
y1[:, 20:108, 20:108, :]=b
y2=np.ones((13434, 128, 128, 1),dtype='float64')
y2[:, 20:108, 20:108, 0]=b[:, :, :, 0]
y1[:, :, :, 0]=y2[:,:,:,0]
np.save('./ydata_128x128_0123_onehot', y1)
