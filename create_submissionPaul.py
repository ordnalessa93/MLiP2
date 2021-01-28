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

def CreateSubmissionPaul(model, test_dir):
    
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
    for x_id, x_name in enumerate(list_x_test[:5]):
        
        # load the image and padd it
        file_x_original = np.array(cv2.imread(os.path.join(test_dir, list_x_test[x_id]), -1))/255
        padded = np.pad(file_x_original, ((0,10),(0,8),(0,0)), 'constant', constant_values = 0)
        
        
        padded = padded[:416, :416, :]
        print(padded.shape)
        print("Starting prediction")
        
        # obtain predicted labels from original image
        file_y_pred = model.predict(np.expand_dims(padded, axis=0))
                
        print("Number of predicted images: {} out of {}".format(x_id, len(list_x_test)))
        
        # once predicted delete the padd
        #file_y_pred = file_y_pred[0, 0:-10, 0:-8]

        
        # get the labels and the confidence of each one
        output_2D = np.array(np.argmax(file_y_pred, axis=-1))
        conf_2D   = np.array(np.amax(file_y_pred, axis=-1))
        output_1D = output_2D.ravel()
        conf_1D   = conf_2D.ravel()
        
        o2D = np.array(output_2D[0,:,:] )
       
        opening = o2D.copy
        ndimage.binary_opening(opening).astype(np.int)
       
        closing = opening.copy
        ndimage.binary_closing(closing).astype(np.int)    
        
        
        plt.figure(figsize = (15,12))
        plt.subplot(1,4,1)
        plt.imshow(padded); plt.title('original_color')
        plt.subplot(1,4,2)
        plt.imshow(o2D); plt.title('predict')
        plt.subplot(1,4,3)
        plt.imshow(opening); plt.title('opening')
        plt.subplot(1,4,4)
        plt.imshow(closing); plt.title('closing')
        plt.show()
        
        
        
        # group all the labels
        groups = groupby(output_1D)
        
        # dictionary containing all the necessary information
        values = {}
        counter = 0

        # iterate over the grouped lists
        for key, group in groups:
            
            group_len = len(list(group))
            print(group_len)
                
            if key != 7:
                
                # add values to the dictionary
                if key not in values:
                    values[key] = [(counter, group_len, group_len, np.mean(conf_1D[counter:counter + group_len - 1]))]      
                else:
                    values[key].append((counter, group_len, group_len, np.mean(conf_1D[counter:counter + group_len - 1])))
                    
            # update the counter
            counter = counter + group_len

        # iterate over the dictionary
        for key in values:

            # append firstly the image id
            image_ids.append(x_name[:-4])

            # then append the corresponding label
            label_ids.append(class_labels[key])

            # variables needed to generate the output
            enc_pix = ''
            count   = 0
            conf    = 0

            # all the values are calculated
            for value in values[key]:
                enc_pix = enc_pix + str(value[0]) + ' ' + str(value[1]) + '|'
                count = count + value[2]
                conf = conf + value[3]

            # get the right confidence
            conf = conf/len(values[key])

            # append the values to the list
            pixel_counts.append(count)
            confidences.append(conf)
            encod_pixels.append(enc_pix)

    # generate the csv
    pd_columns = ['ImageId', 'LabelId', 'PixelCount', 'Confidence', 'EncodedPixels']

    # generate a pandas dataframe with the desired colums
    df = pd.DataFrame(columns = pd_columns)

    # stack the values of the lists into the dataframe
    df['ImageId']       = image_ids
    df['LabelId']       = label_ids
    df['PixelCount']    = pixel_counts
    df['Confidence']    = confidences
    df['EncodedPixels'] = encod_pixels

    print(df)

    # export values to a ".csv" file
    df.to_csv('submitPaul.csv', index=False)