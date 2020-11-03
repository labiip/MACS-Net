#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

def plotSamplesOneHots(labels_of_samples, output_file=False):
    '''
    labels_of_samples of shape (num_samples, x, y, num_onehots)
    '''
    if len(labels_of_samples.shape) != 4:
        print("Incorrect input size - should be (num_samples, x, y, num_onehots)")
    num_samples = labels_of_samples.shape[0]
    num_onehots = labels_of_samples.shape[-1]
    figure_size = (4*num_onehots, 4*num_samples)
    fig, ax = plt.subplots(num_samples, num_onehots, sharex=True, sharey=True, figsize=figure_size)
    for i in range(num_samples):
        for j in range(num_onehots):
            ax[i, j].imshow(labels_of_samples[i,...,j], aspect="auto")
    fig.tight_layout()
    plt.show()
    if output_file == True:
        fig.savefig(output_file)



def oneHotEncode(initial_array):
    '''
    One hot encode the labels
    '''
    allowed_max_class_num = 3
    output_shape = list(initial_array.shape)
    output_shape[-1] = initial_array.max()
    output_array_dims = list(initial_array.shape)
    output_array_dims.append(4)
    output_array = np.zeros(output_array_dims)
    for image_i in range(0, initial_array.shape[0]):
        for class_num in range(0, allowed_max_class_num):
            for x in range(0, initial_array.shape[1]):
                for y in range(0, initial_array.shape[2]):
                    if initial_array[image_i, x, y] == class_num:
                        output_array[image_i, x, y, class_num] = 1

        class_num = allowed_max_class_num
        for x in range(0, initial_array.shape[1]):
            for y in range(0, initial_array.shape[2]):
                if initial_array[image_i, x, y] >= allowed_max_class_num:
                    output_array[image_i, x, y, class_num] = 1
    return output_array


def findNearestNeighbourLabel(array):
    center = int(array.shape[0]/2)
    labels_count = np.zeros(5)
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            if (x != center) or (y != center):
                temp_label = array[x, y]
                labels_count[temp_label] += 1
    return labels_count.argmax()
    

def cleanLabelNearestNeighbour(label):
    '''
    Corrects incorrect labels in a single image based on a threshold on the number of 
    nearest neighbours with the same label
    '''
    x_length = label.shape[0]
    y_length = label.shape[1]
    num_of_classes = 4
    cleaned_labels = np.zeros((x_length, y_length, 4))
    for x in range(1,x_length-1):
        for y in range(1, y_length-1):
            temp_label = label[x,y]
            if temp_label >3: # if labeled as 4 or above
                temp_label = findNearestNeighbourLabel(label[(x-1):(x+2), (y-1):(y+2)])
                cleaned_labels[x, y, temp_label] = 1
            elif temp_label > 0:
                num_labels_in_3x3 = len(np.where(label[(x-1):(x+2), (y-1):(y+2)]==temp_label)[0])
                if num_labels_in_3x3 > 3:
                    cleaned_labels[x, y, temp_label] = 1
                else:
                    temp_label = findNearestNeighbourLabel(label[(x-1):(x+2), (y-1):(y+2)])
                    cleaned_labels[x, y, temp_label] = 1
        non_zero_array = cleaned_labels[..., 1:].sum(axis=2).astype('bool')
        cleaned_labels[..., 0] = np.ones((x_length, y_length), dtype='bool')^non_zero_array
    return cleaned_labels

def cleanLabelNearestNeighbour_alllabels(labels):    
    '''
    Cleans incorrect labels
    '''
    num_labels = labels.shape[0]
    num_of_classes = 4
    cleaned_dim = list(labels.shape)
    cleaned_dim.append(num_of_classes)
    cleaned_labels = np.zeros(cleaned_dim)
    for image_i in range(num_labels):
        print('Preprocessing image %d of %d' % (image_i, num_labels))
        cleaned_labels[image_i,...] = cleanLabelNearestNeighbour(labels[image_i, ...])
    return cleaned_labels


def label012Chromosomes(labels):
    '''
    Input array of (num_samples, x, y, 4)
    Returns array of (num_samples, x, y, 3) where chromosome A and chromosome B are merged
    '''
    labels[...,1] = labels[...,1:3].sum(axis=-1)
    return labels[...,[0,1,3]]
    

def makeXbyY(data, X, Y):
    '''
    Crop data to size X by Y
    '''
    if len(data.shape) < 3:
        print('Input should be of size (num_samples, x, y,...)')
    data_x_start = int((X-data.shape[1])/2)
    data_y_start = int((Y-data.shape[1])/2)
    arrayXbyY = data[:, (data_x_start):(data_x_start + X), (data_y_start):(data_y_start + Y),...]
    return arrayXbyY

def meanIOU_per_image(y_pred, y_true):
    '''
    Calculate the IOU, averaged across images
    '''
    if len(y_pred.shape) < 3 or (y_pred.shape[2]<4):
        print('Wrong dimensions: one hot encoding expected')
        return
    y_pred = y_pred.astype('bool')
    y_true = y_true.astype('bool')
    IUs = []
    for layer in range(y_true.shape[2]):
        intersection = y_pred[...,layer] & y_true[...,layer]
        union = y_pred[...,layer] | y_true[...,layer]
        if union.sum() == 0:
            IUs.append(1)
        else:
            IUs.append(intersection.sum()/union.sum())
    return sum(IUs)/len(IUs)

def meanIOU(y_pred, y_true):
    '''
    Calculate the mean IOU, with the mean taken over classes
    '''
    if len(y_pred.shape) < 4 or (y_pred.shape[3]<4):
        print('Wrong dimensions: one hot encoding expected')
        return
    y_pred = y_pred.astype('bool')
    y_true = y_true.astype('bool')
    IUs = []
    for layer in range(y_true.shape[3]):
        intersection = y_pred[...,layer] & y_true[...,layer]
        union = y_pred[...,layer] | y_true[...,layer]
        if union.sum() == 0:
            IUs.append(1)
        else:
            IUs.append(intersection.sum()/union.sum())
    return sum(IUs)/len(IUs)
	
def IOU(y_pred, y_true):
    '''
    Calculate the IOU for each class seperately
    '''
    if len(y_pred.shape) < 4 or (y_pred.shape[3]<4):
        print('Wrong dimensions: one hot encoding expected')
        return
    y_pred = y_pred.astype('bool')
    y_true = y_true.astype('bool')
    IUs = []
    for layer in range(y_true.shape[3]):
        intersection = y_pred[...,layer] & y_true[...,layer]
        union = y_pred[...,layer] | y_true[...,layer]
        if union.sum() == 0:
            IUs.append(1)
        else:
            IUs.append(intersection.sum()/union.sum())
    return IUs

def IOU_One(y_pred, y_true):
    '''
    Calculate the IOU for each class seperately
    '''
    if len(y_pred.shape) < 4 or (y_pred.shape[3]<4):
        print('Wrong dimensions: one hot encoding expected')
        return  
    IUs = []
    for layer in range(2):
        gt_chrom = y_true[...,layer+1]+y_true[...,3]
        gt_chrom = gt_chrom.astype('bool')
        dr_chrom = y_pred[...,layer+1]+y_pred[...,3]
        dr_chrom = dr_chrom.astype('bool')
        intersection = dr_chrom & gt_chrom
        union = dr_chrom | gt_chrom
        if union.sum() == 0:
            IUs.append(1)
        else:
            IUs.append(intersection.sum()/union.sum())
    return IUs

def IOU_set(y_pred, y_true):
    '''
    Calculate the IOU for each class seperately
    '''
    if len(y_pred.shape) < 4 or (y_pred.shape[3]<4):
        print('Wrong dimensions: one hot encoding expected')
        return
    y_pred = y_pred.astype('bool')
    y_true = y_true.astype('bool')
    IUs = []
    IOU_set = []
    for i in range(y_true.shape[0]):
        intersection = y_pred[i,:,:,3] & y_true[i,:,:,3]
        union = y_pred[i,:,:,3] | y_true[i,:,:,3]
        if union.sum() == 0:
            IUs = 1
        else:
            IUs = intersection.sum()/union.sum()
        IOU_set.append(IUs)
    return IOU_set
    

def globalAccuracy(y_pred, y_true):
    '''
    Calculate the global accuracy (ie. percent of pixels correctly labelled)
    '''
    y_pred = y_pred.astype('bool')
    y_true = y_true.astype('bool')
    correct = y_pred & y_true
    num_correct = correct.sum()
    num_total = 1
    for dim in y_true.shape[0:-1]:
        num_total = num_total*dim
    return num_correct/num_total
    
def global_chrom_Accuracy(y_pred, y_true):
    '''
    Calculate the global accuracy (ie. percent of pixels correctly labelled)
    '''
    y_pred = y_pred.astype('bool')
    y_true = y_true.astype('bool')
    num_total = 1
    for dim in y_true.shape[0:-1]:
        print(dim)
        num_total = num_total*dim
    num_correct = []
    for layer in range(2):
        gt_chrom = y_true[...,layer+1]+y_true[...,3]
        gt_chrom = gt_chrom.astype('bool')
        dr_chrom = y_pred[...,layer+1]+y_pred[...,3]
        dr_chrom = dr_chrom.astype('bool')
        print(gt_chrom.shape)
        num_correct.append(np.equal(gt_chrom, dr_chrom).sum()/num_total)
    return num_correct
    
    
    
