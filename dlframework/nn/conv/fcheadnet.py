from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense


class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        '''
            parameters
            ----------
            baseModel - the original model with FC chopped off
            classes   - number of classes 
            D         - number of nodes in Fully Connected layer
        '''
        # initialize the head model that will be placed on top of the base, then add a FC layer
        headModel = baseModel.output
        headModel = Flatten(name="Flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        # add a softmax layer
        headModel = Dense(classes, activation="softmax ")(headModel)

        return headModel