import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                              

#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
import os
import math
import utilities
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from Loss_accuracy import LossHistory
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import EarlyStopping

from UNet import UNet
from CENet import CENet
from MACS_Net import MACS_Net

# —————————————— Load data ————————————————————————————
xdata = np.load('./xdata_128x128.npy')  
labels = np.load('./ydata_128x128_0123_onehot.npy')

# —————————————— Load the class of data ——————————————————————
number = 1
a=np.load('./data_cls_new/'+str(number)+'/data_cls_4.npy')            
a=a.tolist()
b=np.load('./data_cls_new/'+str(number)+'/data_cls_1.npy')                 
b=b.tolist()

# —————————————— Divide training set and test set ————————————————
x = xdata[a]
y = labels[a]
x_test = xdata[b]
y_test = labels[b]

# —————————————— Pretrained model ———————————————————————
Name = './h5/MACSNet_1.h5'                                      
#Name = './h5/CENet_1.h5'
#Name = './h5/UNet_1.h5'   

# —————————————— Load model ———————————————————————————
#model = CENet(input_shape=(128,128,1))
#model = UNet(input_shape=(128,128,1))
model = MACS_Net(input_shape=(128,128,1))


# —————————————— Load weight ——————————————————————————
model.load_weights(Name)                                                         

# —————————————— predict single img ———————————————————————
ix = 250
img = x_test[ix,:,:,0].reshape(1,128,128,1)
label = y_test[ix,:,:,3]

img_pred = model.predict(img).round()
plt.xticks(())
plt.yticks(())
plt.imshow(x_test[ix,:,:,0])
plt.savefig('./img.png')
plt.imshow(label)
plt.savefig('./label.png')
plt.show()
plt.imshow(img_pred[0,:,:,3])
plt.savefig('./pred.png')
plt.show()

testIOU = utilities.IOU(img_pred, y_test[ix,:,:,:].reshape(1,128,128,4))
print('Testing IOU: ' + str(testIOU))
# —————————————— predict iou ——————————————————————————
y_pred_test = model.predict(x_test).round()
testIOU = utilities.IOU(y_pred_test, y_test)
print('Testing IOU: ' + str(testIOU))
# —————————————— predict chrom iou ———————————————————————
y_pred_test = model.predict(x_test).round()
testIOU = utilities.IOU_One(y_pred_test, y_test)
print('Testing Chrom IOU: ' + str(testIOU))
# —————————————— predict Accuracy ———————————————————————
y_pred_test = model.predict(x_test).round()
testIOU = utilities.global_chrom_Accuracy(y_pred_test, y_test)
print('Testing Chrom Acc: ' + str(testIOU))



