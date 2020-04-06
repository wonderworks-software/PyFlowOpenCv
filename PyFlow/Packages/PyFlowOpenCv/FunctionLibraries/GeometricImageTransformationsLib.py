import cv2

from PyFlow.Core import (
    FunctionLibraryBase,
    IMPLEMENT_NODE
)
from PyFlow.Core.Common import *

def clamp(value,minV,maxV):
    return min(max(minV,value),maxV)

def oddify(value):
    return max(0,value+(value-1))

class GeometricImageTransformationsLib(FunctionLibraryBase):
    '''doc string for GeometricImageTransformationsLib'''

    def __init__(self, packageName):
        super(GeometricImageTransformationsLib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Transform(  input=('ImagePin', 0), 
                    xCenter=('IntPin', 0), yCenter=('IntPin', 75),
                    angle=('FloatPin',0), scale=('FloatPin', 1),
                    img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        rows,cols = input.image.shape[0],input.image.shape[1]
        M = cv2.getRotationMatrix2D((xCenter,yCenter),angle,scale)
        img(cv2.warpAffine(input.image,M,(cols,rows)))  
 