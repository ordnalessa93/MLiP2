# needed imports
import numpy as np
import os
import cv2
from cv2 import connectedComponents

import pandas as pd

# deep learning library
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, UpSampling2D
from keras.optimizers import Adam
from keras import Model

from itertools import groupby

# plotting images
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.morphology import binary_closing, closing, opening
from skimage.morphology import square

def CreateSubmission(model, test_dir):
    
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
        
        
        with open("current_status_create_submission.txt", "a") as myfile:
            myfile.write("Number of predicted images: {} out of {}".format(x_id, len(list_x_test)))
            
        # load the image and padd it
        file_x_original = np.array(cv2.imread(os.path.join(test_dir, x_name), -1))/255
        padded = np.pad(file_x_original, ((0,10),(0,8),(0,0)), 'constant', constant_values = 0)
        
        print("Starting prediction")
        
        # obtain predicted labels from original image
        file_y_pred = model.predict(np.expand_dims(padded, axis=0))
                
        print("Number of predicted images: {} out of {}".format(x_id, len(list_x_test)))
        
        # once predicted delete the padd
        file_y_pred = file_y_pred[0, 0:-10, 0:-8]

        # get the labels and the confidence of each one
        output_2D = np.array(np.argmax(file_y_pred, axis=-1))
        conf_2D   = np.array(np.amax(file_y_pred, axis=-1))
        
        #print(np.unique(output_2D))
        #print(output_2D.dtype)
        #image = imresize(output_2D, 0.5)
        image = np.copy(output_2D)
        #print(np.unique(image))
        image_aux = np.copy(image)
    
        new_image = np.zeros(image.shape, dtype = np.int32)
        rev_unique = reversed(np.unique(image_aux)[:-1])
        for cls in rev_unique:
            #print(cls)
            img = np.copy(image_aux) 
            img[img == cls] = 255
            img[img < 8] = 0
            image_gray = rgb2gray(img)
            clos = closing(image_gray, square(25))
            op = opening(clos, square(25))
            cc = connectedComponents(op.astype(np.uint8), 4, cv2.CV_32S)
            new_image[cc[1] > 0] = cc[1][cc[1] > 0] + class_labels[cls] * 1000

        #img = imresize(new_image, 2.0)
        
#        plt.figure(figsize = (15,12))
#        plt.subplot(2,3,1)
#        plt.imshow(file_x_original); plt.title('original_color')
#        plt.subplot(2,3,2)
#        plt.imshow(image); plt.title('original_label')
#        plt.subplot(2,3,3)
#        plt.imshow(new_image); plt.title('filtered_label')
#        plt.show()
        
        output_1D = new_image.ravel()
        conf_1D   = conf_2D.ravel()

        # group all the labels
        groups = groupby(output_1D)
        
        # dictionary containing all the necessary information
        values = {}
        #start in -1 so when adding len 20 the index is 19 as wanted
        counter = -1

        # iterate over the grouped lists
        for key, group in groups:

            group_len = len(list(group))
            #print("group_len ",group_len)
                
            if key != 0:# rejecting background
                
                # add values to the dictionary
                if key not in values:
                    values[key] = [(counter+1, group_len, group_len, np.sum(conf_1D[counter+1:counter + group_len]))]      
                else:
                    values[key].append((counter+1, group_len, group_len, np.sum(conf_1D[counter+1:counter + group_len])))
                    
            # update the counter
            counter = counter + group_len
        
        # iterate over the dictionary
        for key in values:

            # append firstly the image id
            image_ids.append(x_name[:-4])

            # then append the corresponding label
            label_ids.append(key // 1000) # class_labels[key]

            # variables needed to generate the output
            enc_pix = ''
            count   = 0
            conf    = 0

            # all the values are calculated
            for value in values[key]:
                enc_pix = enc_pix + str(value[0]) + ' ' + str(value[1]) + '|'
                count = count + value[2]
                conf = conf + value[3]
            #print("enc_pix = ",enc_pix)

            # get the right confidence all the summed confidence of the pixels devided by the number of pixels
            conf = conf/count

            # append the values to the list
            pixel_counts.append(count)
            confidences.append(conf)
            encod_pixels.append(enc_pix)

    # generate the csv
    pd_columns = ['ImageId', 'LabelId', 'Confidence', 'PixelCount', 'EncodedPixels']

    # generate a pandas dataframe with the desired colums
    df = pd.DataFrame(columns = pd_columns)

    # stack the values of the lists into the dataframe
    df['ImageId']       = image_ids
    df['LabelId']       = label_ids
    df['Confidence']    = confidences
    df['PixelCount']    = pixel_counts
    df['EncodedPixels'] = encod_pixels

   # print(df)

    # export values to a ".csv" file
    df.to_csv('submit.csv', index=False)