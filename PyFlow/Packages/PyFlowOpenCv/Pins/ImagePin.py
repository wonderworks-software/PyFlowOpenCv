from PyFlow.Core import PinBase
from PyFlow.Core.Common import *
import numpy as np

class MyImage():
    def __init__(self,image=None):
        if isinstance(image,MyImage):
            self.image = image.image.copy()
        elif isinstance(image,np.ndarray):
            self.image = image
        else:
            self.image = np.zeros((2, 2, 3), np.uint8)

class ImagePin(PinBase):
    """doc string for ImagePin"""
    def __init__(self, name, parent, direction, **kwargs):
        super(ImagePin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue(MyImage())
        self.disableOptions(PinOptions.Storable)

    @staticmethod
    def IsValuePin():
        return True

    @staticmethod
    def supportedDataTypes():
        return ('ImagePin',)

    @staticmethod
    def pinDataTypeHint():
        return 'ImagePin', MyImage()

    @staticmethod
    def color():
        return (200, 200, 50, 255)

    @staticmethod
    def internalDataStructure():
        return MyImage

    @staticmethod
    def processData(data):
        return ImagePin.internalDataStructure()(data)
