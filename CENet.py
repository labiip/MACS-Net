#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D,ZeroPadding2D
from keras.layers.convolutional import AtrousConvolution2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, BatchNormalization, Dropout, add
from keras.layers import concatenate
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras import callbacks
from keras.layers.core import Lambda
from keras.layers import UpSampling2D
import numpy as np
import tensorflow as tf

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x

    
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), num = 1):
    x = inpt 
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) 
    if np.shape(x)[-1] != nb_filter : 
        x = Conv2d_BN(x,nb_filter=nb_filter,kernel_size=(1,1),strides=strides,padding='same')
    for i in range(num):      
        x_1 = Conv2d_BN(x,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
        x_2 = Conv2d_BN(x_1, nb_filter=nb_filter, kernel_size=kernel_size,strides=strides,padding='same')
        x = add([x_2,x])
    return x

def Dense_Atrous_Convolution(x):
    #Branch 1
    x1 = AtrousConvolution2D(512, 3, 3,atrous_rate=(1,1),border_mode='same')(x)
    #Branch 2
    x2_1 = AtrousConvolution2D(512, 3, 3,atrous_rate=(3,3),border_mode='same')(x)
    x2_2 = AtrousConvolution2D(512, 1, 1,atrous_rate=(1,1),border_mode='same')(x2_1)    
    #Branch 3
    x3_1 = AtrousConvolution2D(512, 3, 3,atrous_rate=(1,1),border_mode='same')(x)
    x3_2 = AtrousConvolution2D(512, 3, 3,atrous_rate=(3,3),border_mode='same')(x3_1)
    x3_3 = AtrousConvolution2D(512, 1, 1,atrous_rate=(1,1),border_mode='same')(x3_2)
    #Branch 4
    x4_1 = AtrousConvolution2D(512, 3, 3,atrous_rate=(1,1),border_mode='same')(x) 
    x4_2 = AtrousConvolution2D(512, 3, 3,atrous_rate=(3,3),border_mode='same')(x4_1)
    x4_3 = AtrousConvolution2D(512, 3, 3,atrous_rate=(5,5),border_mode='same')(x4_2)
    x4_4 = AtrousConvolution2D(512, 1, 1,atrous_rate=(1,1),border_mode='same')(x4_3)
    x = add([x1,x2_2,x3_3,x4_4,x])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x

def resize_featuremap(x):
    return tf.compat.v1.image.resize(x,[4,4],method='bilinear', align_corners=True)
def Residual_Multikernel_pooling(x):
    #Maxpooling
    x1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
    x2 = MaxPooling2D(pool_size=(3,3),strides=(3,3),padding='same')(x)
    x3 = MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same')(x)
    x4 = MaxPooling2D(pool_size=(6,6),strides=(6,6),padding='same')(x)
    #1*1 conv
    x1_1 = Conv2D(1, (1,1), activation='relu', padding='same')(x1)
    x2_1 = Conv2D(1, (1,1), activation='relu', padding='same')(x2)
    x3_1 = Conv2D(1, (1,1), activation='relu', padding='same')(x3)
    x4_1 = Conv2D(1, (1,1), activation='relu', padding='same')(x4)
    #upsample    
    x1_2 = Lambda(resize_featuremap)(x1_1)
    x2_2 = Lambda(resize_featuremap)(x2_1)
    x3_2 = Lambda(resize_featuremap)(x3_1)
    x4_2 = Lambda(resize_featuremap)(x4_1)
    #concatenate
    x = concatenate([x1_2,x2_2,x3_2,x4_2,x])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x



def CENet_ceshi(input_tensor=None, input_shape=None, pooling=None):  
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor   
            
    x_64 = Conv2d_BN(img_input,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='same')   
    #(32,32,64)
    x_32 = Conv_Block(x_64,nb_filter=64,kernel_size=(3,3),num = 3)
    #(16,16,128)
    x_16 = Conv_Block(x_32,nb_filter=128,kernel_size=(3,3),num = 4)
    #(8,8,256)
    x_8 = Conv_Block(x_16,nb_filter=256,kernel_size=(3,3),num = 6)
    #(4,4,512)
    x_4 = Conv_Block(x_8,nb_filter=512,kernel_size=(3,3),num = 3)
   
#    #Dense Atrous Convolution module
    x = Dense_Atrous_Convolution(x_4)   
    
#    #Residual Multi-kernel pooling
    x = Residual_Multikernel_pooling(x)
     
    #upsample  
    x = Conv2d_BN(x,nb_filter=512,kernel_size=(1,1),strides=(1,1),padding='same')  
    x = Conv2DTranspose(512,(3, 3),strides=(2, 2),padding='same')(x)  
    x = Conv2d_BN(x,nb_filter=512,kernel_size=(1,1),strides=(1,1),padding='same')  
    x = concatenate([x,x_8])
    
    x = Conv2d_BN(x,nb_filter=256,kernel_size=(1,1),strides=(1,1),padding='same')  
    x = Conv2DTranspose(256,(3, 3),strides=(2, 2),padding='same')(x)  
    x = Conv2d_BN(x,nb_filter=256,kernel_size=(1,1),strides=(1,1),padding='same')  
    x = concatenate([x,x_16])

    x = Conv2d_BN(x,nb_filter=128,kernel_size=(1,1),strides=(1,1),padding='same')  
    x = Conv2DTranspose(128,(3, 3),strides=(2, 2),padding='same')(x)  
    x = Conv2d_BN(x,nb_filter=128,kernel_size=(1,1),strides=(1,1),padding='same')  
    x = concatenate([x,x_32]) 

    x = Conv2d_BN(x,nb_filter=64,kernel_size=(1,1),strides=(1,1),padding='same')  
    x = Conv2DTranspose(64,(3, 3),strides=(2, 2),padding='same')(x)  
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(1,1),strides=(1,1),padding='same')    
    x = concatenate([x,x_64])
    
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(1,1),strides=(1,1),padding='same')  
    x = Conv2DTranspose(64,(3, 3),strides=(2, 2),padding='same')(x)  
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(1,1),strides=(1,1),padding='same')  
    
    x = Conv2D(4, (1, 1), activation='sigmoid', padding='same')(x)


    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model
    model = Model(inputs , x, name='CENet_ceshi')

    return model
