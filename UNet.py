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
import numpy as np



def UNet(input_tensor=None, input_shape=None, pooling=None):
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
    x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x_1)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv3')(x)
    x = BatchNormalization()(x_2)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)


    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    
    # Block 2B
    x = concatenate([x, x_2]) 
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock2_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock2_conv3')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
	
    # Block 1B
    x = concatenate([x, x_1])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dblock1_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dblock1_conv2')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    # Output convolution. Number of filters should equal number of channels of the output
    x = Conv2D(4, (1, 1), activation='sigmoid', padding='same', name='dblock1_conv3')(x)
     


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model
    model = Model(inputs, x, name='UNet')
    model.summary()


    return model
