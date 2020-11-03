import os
#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import warnings
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, BatchNormalization, Dropout,add
from keras.layers.convolutional import AtrousConvolution2D
from keras.layers import concatenate
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras import callbacks
from keras.layers import UpSampling2D
from keras.layers.core import Lambda
from keras.layers import UpSampling2D
import numpy as np
import tensorflow as tf

def MAC(x):
    #Branch 1
    x1 = AtrousConvolution2D(256, 3, 3,atrous_rate=(1,1),border_mode='same')(x)
    #Branch 2
    x2_1 = AtrousConvolution2D(256, 3, 3,atrous_rate=(2,2),border_mode='same')(x)
    x2_2 = AtrousConvolution2D(256, 1, 1,atrous_rate=(1,1),border_mode='same')(x2_1)    
    #Branch 3
    x3_2 = AtrousConvolution2D(256, 3, 3,atrous_rate=(3,3),border_mode='same')(x)
    x3_3 = AtrousConvolution2D(256, 1, 1,atrous_rate=(1,1),border_mode='same')(x3_2)
    #Branch 4
    x4_3 = AtrousConvolution2D(256, 3, 3,atrous_rate=(4,4),border_mode='same')(x)
    x4_4 = AtrousConvolution2D(256, 1, 1,atrous_rate=(1,1),border_mode='same')(x4_3)
    x = add([x1,x2_2,x3_3,x4_4,x])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x 

def SSPM(x):
    #Maxpooling
    x1 = MaxPooling2D(pool_size=(2,2),   strides=(2,2),padding='same')(x)
    x2 = MaxPooling2D(pool_size=(3,3),   strides=(2,2),padding='same')(x)
    x3 = MaxPooling2D(pool_size=(4,4),   strides=(2,2),padding='same')(x)
    x4 = MaxPooling2D(pool_size=(8,8),   strides=(2,2),padding='same')(x)
    #1*1 conv
    x1_1 = Conv2D(1, (1,1), activation='relu', padding='same')(x1)
    x2_1 = Conv2D(1, (1,1), activation='relu', padding='same')(x2)
    x3_1 = Conv2D(1, (1,1), activation='relu', padding='same')(x3)
    x4_1 = Conv2D(1, (1,1), activation='relu', padding='same')(x4)
    #upsample
    x1_2 = UpSampling2D(size=(2, 2))(x1_1)
    x2_2 = UpSampling2D(size=(2, 2))(x2_1)
    x3_2 = UpSampling2D(size=(2, 2))(x3_1)
    x4_2 = UpSampling2D(size=(2, 2))(x4_1)
    #concatenate
    x = concatenate([x1_2,x2_2,x3_2,x4_2,x])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x

def DAC(x):
    #Branch 1
    x1 = AtrousConvolution2D(256, 3, 3,atrous_rate=(1,1),border_mode='same')(x)
    #Branch 2
    x2_1 = AtrousConvolution2D(256, 3, 3,atrous_rate=(3,3),border_mode='same')(x)
    x2_2 = AtrousConvolution2D(256, 1, 1,atrous_rate=(1,1),border_mode='same')(x2_1)    
    #Branch 3
    x3_1 = AtrousConvolution2D(256, 3, 3,atrous_rate=(1,1),border_mode='same')(x)
    x3_2 = AtrousConvolution2D(256, 3, 3,atrous_rate=(3,3),border_mode='same')(x3_1)
    x3_3 = AtrousConvolution2D(256, 1, 1,atrous_rate=(1,1),border_mode='same')(x3_2)
    #Branch 4
    x4_1 = AtrousConvolution2D(256, 3, 3,atrous_rate=(1,1),border_mode='same')(x) 
    x4_2 = AtrousConvolution2D(256, 3, 3,atrous_rate=(3,3),border_mode='same')(x4_1)
    x4_3 = AtrousConvolution2D(256, 3, 3,atrous_rate=(5,5),border_mode='same')(x4_2)
    x4_4 = AtrousConvolution2D(256, 1, 1,atrous_rate=(1,1),border_mode='same')(x4_3)
    x = add([x1,x2_2,x3_3,x4_4,x])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x

def resize_featuremap(x):
    return tf.compat.v1.image.resize(x,[16,16],method='bilinear', align_corners=True)
def PPM(x):
    #Maxpooling
    x1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
    x2 = MaxPooling2D(pool_size=(3,3),strides=(3,3),padding='same')(x)
    x3 = MaxPooling2D(pool_size=(4,4),strides=(4,4),padding='same')(x)
    x4 = MaxPooling2D(pool_size=(8,8),strides=(8,8),padding='same')(x)
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

def Res_Path(x,nb_filter=[32,8],strides=(1,1)):
    for i in range(0,nb_filter[1],2):
        x_1 = Conv2D(nb_filter[0], (3, 3), activation='relu',strides=strides,padding='same')(x)
        x_2 = Conv2D(nb_filter[0], (1, 1), activation='relu',strides=strides,padding='same')(x_1)
        x = add([x_1,x_2])
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    return x

def MACS_Net(input_tensor=None, input_shape=None, pooling=None):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x_1a = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x_1a)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x_2a = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv3')(x)
    x = BatchNormalization()(x_2a)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)
    x = Dropout(0.5)(x)


    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x_3a = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x_3a)
    x = MaxPooling2D((2, 2),name='block3_pool')(x)
    x = Dropout(0.5)(x)
    
    x = MAC(x)
    #x = DAC(x)
    
    x = SSPM(x)
    #x = PPM(x)
      
    ####################################################################################################################
    x_3b = UpSampling2D(size=(2, 2))(x)	
    # Block 3B
    x_3a = Res_Path(x_3a,nb_filter=[128,4],strides=(1,1))
    x = concatenate([x_3a, x_3b])  
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock8_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock8_conv2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock8_conv3')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x_2b = UpSampling2D(size=(2, 2))(x)
    
    # Block 2B
    x_2a = Res_Path(x_2a,nb_filter=[64,6],strides=(1,1))
    x = concatenate([x_2a, x_2b]) 
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock9_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock9_conv2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock9_conv3')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x_1b = UpSampling2D(size=(2, 2))(x)
	
    # Block 1B
    x_1a = Res_Path(x_1a,nb_filter=[32,8],strides=(1,1))
    x = concatenate([x_1a, x_1b])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dblock10_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dblock10_conv2')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    # Output convolution. Number of filters should equal number of channels of the output
    x = Conv2D(4, (1, 1), activation='sigmoid', padding='same', name='dblock10_conv3')(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model
    model = Model(inputs, x, name='MACS_Net')
    model.summary()


    return model
