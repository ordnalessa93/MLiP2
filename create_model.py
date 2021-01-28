from keras import Model
from keras.layers import *
from keras.applications import vgg16, resnet50
import keras.backend as K
from keras.utils import get_file 
from keras.layers.core import SpatialDropout2D, Activation

def createVGG16(in_t, weights='imagenet'):

    model = vgg16.VGG16(include_top = False, weights = 'imagenet', input_tensor = in_t)
    
    # freeze the layers
    for layer in model.layers:
        layer.trainable = False
    
    # get the output of the model
    x = model.layers[-1].output

    # 8 filters of 3x3
    x = Conv2DTranspose(filters = 256,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               activation = 'relu',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    x = Dropout(0.2)(x)
    
    # 8 filters of 3x3
    x = Conv2DTranspose(filters = 128,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               activation = 'relu',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    x = Dropout(0.2)(x)

    # 8 filters of 3x3
    x = Conv2DTranspose(filters = 64,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               activation = 'relu',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    x = Dropout(0.2)(x)

    # 8 filters of 3x3
    x = Conv2DTranspose(filters = 32,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               activation = 'relu',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    x = Dropout(0.2)(x)
    
    # 8 filters of 3x3
    x = Conv2DTranspose(filters = 8,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               activation = 'relu',
               padding = 'same')(x)

    x = UpSampling2D(size = (2, 2))(x)

    # dropout to avoid overfitting
    x = Dropout(0.3)(x) 

    #x = UpSampling2D(size = (32, 32))(x)

    x_out = Activation(activation = 'softmax')(x)

    vgg_16_new = Model(input = in_t, output = x_out)
    vgg_16_new.summary()
    return vgg_16_new


def createVGGskip(in_t, num_class):
    
    # since we are going to segment the image, the input should not have any dimension

    vgg_16 = vgg16.VGG16(weights='imagenet', include_top=False,  input_tensor = in_t)

    for l in vgg_16.layers:
        l.trainable = False


    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    # CONV 0

    x = Conv2D(filters = 512,
               kernel_size = (3,3),
               kernel_initializer = 'he_uniform',
               padding = 'same')(vgg_16.layers[-1].output)
    
    x = BatchNormalization()(x)

    x = Activation(activation = 'relu')(x)

    x = Dropout(0.2) (x)

    # CONV + UPSAMPLING 1

    x = UpSampling2D(size = (2,2 ))(x)

    x = concatenate([x,vgg_16.get_layer("block5_conv3").output])

    x = Conv2D(filters = 256,
               kernel_size = (3,3),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)
    

    x = BatchNormalization()(x)

    x = Activation(activation = 'relu')(x)

    x = Dropout(0.2) (x)

    # CONV + UPSAMPLING 2

    x = UpSampling2D(size = (2, 2))(x)

    x = concatenate([x,vgg_16.get_layer("block4_conv3").output])

    x = Conv2D(filters = 128,
               kernel_size = (3,3),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)
    

    x = BatchNormalization()(x)

    x = Activation(activation = 'relu')(x)

    x = Dropout(0.2) (x)

    # CONV + UPSAMPLING 3

    x = UpSampling2D(size = (2, 2))(x)

    x = concatenate([x,vgg_16.get_layer("block3_conv3").output])

    x = Conv2D(filters = 64,
               kernel_size = (3,3),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)

    x = BatchNormalization()(x)

    x = Activation(activation = 'relu')(x)

    x = Dropout(0.2)(x)

    # CONV + UPSAMPLING 4

    x = UpSampling2D(size = (2, 2))(x)

    x = concatenate([x,vgg_16.get_layer("block2_conv2").output])

    x = Conv2D(filters = 32,
               kernel_size = (3,3),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)

    x = BatchNormalization()(x)

    x = Activation(activation = 'relu')(x)

    x = Dropout(0.2) (x)

    # CONV + UPSAMPLING 5

    x = UpSampling2D(size = (2, 2))(x)

    x = concatenate([x,vgg_16.get_layer("block1_conv2").output])

    x = Conv2D(filters = num_class,
               kernel_size = (3,3),
               kernel_initializer = 'he_uniform',
               padding = 'same')(x)

    x = BatchNormalization()(x)

    x = Activation(activation = 'relu')(x)

    x = Dropout(0.2)(x)

    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------

    x_out = Activation(activation = 'softmax')(x)

    vgg_16_new = Model(input = in_t, output = x_out)

    vgg_16_new.summary()
    
    return vgg_16_new

# ------------ Unet Pre-Trained ---------------

# Pretrained weights
ZF_UNET_224_WEIGHT_PATH = 'https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/releases/download/v1.0/zf_unet_224.h5'

def double_conv_layer(x, size):

    axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    conv = BatchNormalization(axis=axis)(conv)
    
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    
    conv = BatchNormalization(axis=axis)(conv)
    
    conv = Activation('relu')(conv)
    conv = SpatialDropout2D(0.2)(conv)
    return conv

def createUnet(x_in):

    inputs = x_in #Input((224, 224, 3))
    axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters)

    conv_final = Conv2D(1, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224")

    weights_path = get_file(
        'zf_unet_224_weights_tf_dim_ordering_tf_generator.h5',
        ZF_UNET_224_WEIGHT_PATH,
        cache_subdir='models',
        file_hash='203146f209baf34ac0d793e1691f1ab7')
    model.load_weights(weights_path)
    
    model.layers.pop()
    model.layers.pop()
    
    for layer in model.layers[:-8]:
        layer.trainable = False
    
    x = model.layers[-1].output
    
    conv_final = Conv2D(8, (1, 1))(x)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(model.input, conv_final)
    
    return 



def createResNet50(in_t):

    model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor = in_t)
    
    # freeze the layers
    for layer in model.layers:
        layer.trainable = False
    
    # get the output of the model
    x = model.layers[-2].output

    # 8 filters of 3x3
    x = Conv2D(filters = 256,
               kernel_size = (2,2),
               kernel_initializer = 'he_uniform',
               activation = 'relu',
               padding = 'same')(x)
    
    x_out = Activation(activation = 'softmax')(x)

    ResNet_new = Model(input = in_t, output = x_out)
    ResNet_new.summary()
    return ResNet_new