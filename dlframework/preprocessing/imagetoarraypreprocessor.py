from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:

    def __init__(self, dataFormat = None):

        self.dataFormat = dataFormat #indicates channel ordering, None indicates the keras.json settings to be used

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image
        return img_to_array(image, data_format = self.dataFormat)