# needed imports
import numpy as np
import os
import cv2
import pandas as pd

# deep learning library
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, UpSampling2D
from keras.optimizers import Adam
from keras import Model

import scipy
from scipy import ndimage

from itertools import groupby

# plotting images
import matplotlib.pyplot as plt

def MakePrediction(model, test_dir):
    
    # obtain a list of files in the test directory
    list_x_test = np.array(sorted(os.listdir(test_dir)))
    
    # numpy containing the labels
    class_labels = np.array((33, 34, 35, 36, 38, 39, 40, 0))
        
    # lists containing all the values
    image_ids    = []
    label_ids    = []
    pixel_counts = []
    confidences  = []
    encod_pixels = []
    
    # for each test file
    for x_id, x_name in enumerate(list_x_test):
        
        # load the image and padd it
        file_x_original = np.array(cv2.imread(os.path.join(test_dir, list_x_test[x_id]), -1))/255
        padded = np.pad(file_x_original, ((0,10),(0,8),(0,0)), 'constant', constant_values = 0)
   
        print("Starting prediction")
        
        # obtain predicted labels from original image
        file_y_pred = model.predict(np.expand_dims(padded, axis=0))
                
        print("Number of predicted images: {} out of {}".format(x_id, len(list_x_test)))
        
        # once predicted delete the pad
        file_y_pred = file_y_pred[0, 0:-10, 0:-8]
        
        destination_dir = './data/full_data/predictions'
        filename = os.path.join(destination_dir, list_x_test[x_id][:-4])
        filename += '.png'
        cv2.imwrite(filename, np.argmax(file_y_pred, axis = -1))