from keras import Model
from keras.layers import *
from keras.applications import vgg16
import keras.backend as K
import numpy as np
import cv2, os


class PatchGenerator(object):

    def __init__(self, input_dir, filelist_x, filelist_y, class_labels, batch_size, augmentation_fn=None):

        # params
        self.input_dir    = input_dir   # path to patches in glob format
        self.filelist_x   = filelist_x  # list containing filenames of training
        self.filelist_y   = filelist_y  # list containing filenames of labels
        self.class_labels = class_labels # list containing the classes
        self.batch_size   = batch_size  # number of patches per batch
        
        # info
        self.n_samples = len(self.filelist_x)
        self.n_batches = self.n_samples // self.batch_size
        
        # print some info
        print('Patch Generator with {n_samples} patch samples.'.format(n_samples = self.n_samples))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        # number of batches
        return self.n_batches

    def next(self):
        #print(GPUtil.showUtilization())
                
        # randomly sample indexes from the list
        indxs = np.random.choice(np.arange(self.n_samples), self.batch_size, replace = False)
        
        # maximum pixel size of width and height
        max_width  = 120
        max_height = 120
        
        # batch_x = np.zeros((self.batch_size, 2710, 3384, 3), dtype = np.float32)
        # batch_y = np.zeros((self.batch_size, 2710, 3384, len(class_labels)), dtype = np.int8)
        batch_x = np.random.rand(self.batch_size, max_height, max_width, 3)
        batch_y = np.zeros((self.batch_size, max_height, max_width, len(self.class_labels)))
        
        for i, indx in enumerate(indxs):
            # load the image file
            img = np.array(cv2.imread(os.path.join(self.input_dir, 'train_color', self.filelist_x[indx]), -1))/255
            
            labels = np.zeros((max_height, max_width))
            
            # load the label file
            labels_np = np.array(cv2.imread(os.path.join(self.input_dir, 'train_label', self.filelist_y[indx]), -1))//1000

            labels[:img.shape[0],:img.shape[1]] = labels_np[:max_height,:max_width]
            
            # filter the labels
            labels[np.logical_not(np.isin(labels, self.class_labels[0:-1]))] = 0

            # encode in one_hot
            labels = (self.class_labels == labels[...,None]).astype(int)
            
            # store the image and the labels
            batch_x[i,:img.shape[0],:img.shape[1]] = img[:max_height,:max_width]
            batch_y[i] = labels

        return batch_x, batch_y

    
def createVGG16(in_t, weights='imagenet'):

    model = vgg16.VGG16(include_top = False, weights = 'imagenet', input_tensor = in_t)

    # get the output of the model
    x = model.layers[-1].output

    # 8 filters of 3x3
    x = Conv2D(filters = 8,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)

    x = Activation(activation = 'relu')(x)

    x = UpSampling2D(size = (2, 2))(x)

    # 8 filters of 3x3
    x = Conv2D(filters = 8,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    # 8 filters of 3x3
    x = Conv2D(filters = 8,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    # 8 filters of 3x3
    x = Conv2D(filters = 8,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    # 8 filters of 3x3
    x = Conv2D(filters = 8,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    # dropout to avoid overfitting
    x = Dropout(0.3)(x) 

    #x = UpSampling2D(size = (32, 32))(x)

    x_out = Activation(activation = 'softmax')(x)

    vgg_16_new = Model(input = in_t, output = x_out)
    vgg_16_new.summary()
    return vgg_16_new

def load_image(main_dir, file_name):
    x_path = os.path.join(main_dir, 'train_color', file_name)
    y_path = os.path.join(main_dir, 'train_label', file_name)
    
    if not os.path.exists(x_path):
        raise Exception('Color image file does not exist!')
    if not os.path.exists(y_path):
        raise Exception('Labels file does not exist!')
    
    # load the image file
    img = np.array(cv2.imread(x_path, -1))/255

    # load the label file
    labels_np = np.array(cv2.imread(os.path.join(y_path, -1)))//1000
                         
    return img, labels_np