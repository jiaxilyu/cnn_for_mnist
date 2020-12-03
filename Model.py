from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import numpy as np

 #filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        #When using this layer as the first layer in a model, provide the keyword 
        #argument input_shape (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last"."""
        #kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window."""
        #strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions
        #strides:forward step size in x,y, if (2,1) f(01 02) then f(03 04)
        #if (1,1) f(01 02) then f(02 03)


#define the first cnn network
#a cnn network with 2 convolution layer with max pool
#1st con layer with a 5x5 kernel matrix
#max pool, pooling max form each 2x2 matrix, strides(2,2)
#2nd con layer with a 5x5 kernel matrix and using max pool, strides(2,2)
#max pool, strides(2,2)
#flattern the con layer output
#dense layer with 512 hidden units
#output layer, choose softmax, each output reprsent the possibility of cooresponding digit
def build_cnn1(input_shape, output_size):
    model = Sequential()
    model.add(Conv2D(filters= 32, input_shape=input_shape, kernel_size=(5,5),strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(filters= 64, kernel_size=(5,5), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1)))
    #here is nn
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(output_size,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model