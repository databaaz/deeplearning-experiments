
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

from keras import backend as K


class MiniVGGNet:
    @staticmethod
    def build( width, height, depth, classes):
        
        #initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        channelDim = -1
        
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            channelDim = 1
            
        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3,3), padding="same",
                        input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channelDim))
        model.add(Conv2D(32, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channelDim))
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channelDim))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model
                  


