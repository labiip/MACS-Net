# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:54:04 2019

@author: 95370
"""
import numpy as np 
import math
from matplotlib import pyplot as plt
import cv2
from sklearn.cluster import KMeans
import random
import skimage


def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:    
                output[i][j] = 0 
            elif rdn > thres:  
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output




data = np.load('./xdata_128x128.npy')

#------------------------------ data_sp ------------------------------#  
data2 = np.zeros(shape=(13434,128,128,1))

for i in range(13434):
    print(i)
    data2[i,:,:,:] = sp_noise(data[i,:,:,:],0.01)
np.save('xdata_128x128_sp_0.01.npy',data2)


#------------------------------ data_gaussian ------------------------------#  
data2 = np.zeros(shape=(13434,128,128,1))
data = data/255
for i in range(13434):
    print(i)
    img_noise=skimage.util.random_noise(data[i,:,:,0], mode='gaussian', seed=None, var=(1/255.0)**2)
    data2[i,:,:,0]=img_noise*255 
np.save('xdata_128x128_gaussian_1.npy',data2)


#------------------------------ data_poisson ------------------------------# 
data2 = np.zeros(shape=(13434,128,128,1))
for i in range(13434):
    print(i)
    img_noise=skimage.util.random_noise(data[i,:,:,0], mode='poisson', seed=None,clip=True)
    data2[i,:,:,0]= img_noise
np.save('xdata_128x128_poisson.npy',data2)