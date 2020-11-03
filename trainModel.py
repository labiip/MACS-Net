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

# —————————————— Temporary model ————————————————————————
Name_tem = './MACSNet_'+str(number)+'.h5'             

# —————————————— Load model ———————————————————————————
#model = CENet(input_shape=(128,128,1))
#model = UNet(input_shape=(128,128,1))
model = MACS_Net(input_shape=(128,128,1))



#model = multi_gpu_model(model, gpus=2)                          
model.compile(loss='binary_crossentropy', optimizer='adam')  # Initial learning rate:0.001
model.load_weights(Name)                                                         

    

# —————————————— Specify the number of epochs to run ——————————————
num_epoch = 1                                                                                                                                     
for i in range(num_epoch):
    print('epoch:', i)
    # Fit
    check_point = ModelCheckpoint(Name_tem, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)  
    callback = EarlyStopping(monitor="val_loss", patience=30, verbose=0, mode='min')  
    
    history = model.fit(x, y, epochs=300, validation_split=0.2, batch_size=32, callbacks=[check_point, callback])       
                                       
    # Calculate overlap IOU
    model.load_weights(Name_tem) 
    y_pred_train = model.predict(x).round()
    trainIOU = utilities.IOU(y_pred_train, y)
    print('Training IOU: ' + str(trainIOU))    
    y_pred_test = model.predict(x_test).round()
    testIOU = utilities.IOU(y_pred_test, y_test)
    print('Testing Overlap IOU: ' + str(testIOU))
