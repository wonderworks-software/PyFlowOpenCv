import cv2

from PyFlow.Core import (
    FunctionLibraryBase,
    IMPLEMENT_NODE
)
from PyFlow.Core.Common import *

def oddify(value):
    return max(0,value+(value-1))

class GeometricImageTransformationsLib(FunctionLibraryBase):
    '''doc string for GeometricImageTransformationsLib'''

    def __init__(self, packageName):
        super(GeometricImageTransformationsLib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Transformations', NodeMeta.KEYWORDS: []})
    def image_resize_by_Factor(input=('ImagePin', None), img=(REF, ('ImagePin', None)),
                       factor=('FloatPin', 1.0 ), interpolation=('StringPin', "INTER_CUBIC", {PinSpecifiers.VALUE_LIST: ["INTER_LINEAR","INTER_CUBIC","INTER_AREA","INTER_LANCZOS4","INTER_LINEAR_EXACT","INTER_MAX","WARP_FILL_OUTLIERS","WARP_INVERSE_MAP" ]} )
                       ):    
        inter = {   "INTER_NEAREST": cv2.INTER_NEAREST ,
                            "INTER_LINEAR": cv2.INTER_LINEAR  ,
                            "INTER_CUBIC": cv2.INTER_CUBIC   ,
                            "INTER_AREA": cv2.INTER_AREA   ,
                            "INTER_LANCZOS4": cv2.INTER_LANCZOS4   ,
                            "INTER_LINEAR_EXACT": cv2.INTER_LINEAR_EXACT    ,
                            "INTER_MAX": cv2.INTER_MAX,
                            "WARP_FILL_OUTLIERS": cv2.WARP_FILL_OUTLIERS,
                            "WARP_INVERSE_MAP ": cv2.WARP_INVERSE_MAP
                        }
        imagen = input
        width  = int(imagen.shape[1] * factor)
        height = int(imagen.shape[0] * factor)
        dim = (width, height)
        # resize image
        img( cv2.resize(imagen, dim, interpolation = inter[interpolation]))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Transformations', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Transform(  input=('ImagePin', None), 
                    xCenter=('IntPin', 0), yCenter=('IntPin', 75),
                    angle=('FloatPin',0), scale=('FloatPin', 1),
                    img=(REF, ('ImagePin', None))):
        """ Transform an Image."""
        rows,cols = input.shape[0],input.shape[1]
        M = cv2.getRotationMatrix2D((xCenter,yCenter),angle,scale)
        img(cv2.warpAffine(input,M,(cols,rows)))  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Transformations', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_FlipImage(input=('ImagePin', None), mode=('IntPin', 0), img=(REF, ('ImagePin', None))):
        """Return a frame of the loaded image."""
        img(cv2.flip(input, mode))        