import cv2

from PyFlow.Core import (
    FunctionLibraryBase,
    IMPLEMENT_NODE
)
from PyFlow.Core.Common import *
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import MyImage

def oddify(value):
    return max(0,value+(value-1))

class GeometricImageTransformationsLib(FunctionLibraryBase):
    '''doc string for GeometricImageTransformationsLib'''

    def __init__(self, packageName):
        super(GeometricImageTransformationsLib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Transformations', NodeMeta.KEYWORDS: []})
    def image_resize_by_Factor(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                       factor=('FloatPin', 1.0 ), interpolation=('StringPin', "INTER_CUBIC", {PinSpecifires.VALUE_LIST: ["INTER_LINEAR","INTER_CUBIC","INTER_AREA","INTER_LANCZOS4","INTER_LINEAR_EXACT","INTER_MAX","WARP_FILL_OUTLIERS","WARP_INVERSE_MAP" ]} )
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
        imagen = input.image
        width  = int(imagen.shape[1] * factor)
        height = int(imagen.shape[0] * factor)
        dim = (width, height)
        # resize image
        img( cv2.resize(imagen, dim, interpolation = inter[interpolation]))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Transformations', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Transform(  input=('ImagePin', 0), 
                    xCenter=('IntPin', 0), yCenter=('IntPin', 75),
                    angle=('FloatPin',0), scale=('FloatPin', 1),
                    img=(REF, ('ImagePin', 0))):
        """ Transform an Image."""
        rows,cols = input.image.shape[0],input.image.shape[1]
        M = cv2.getRotationMatrix2D((xCenter,yCenter),angle,scale)
        img(cv2.warpAffine(input.image,M,(cols,rows)))  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Transformations', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_FlipImage(input=('ImagePin', 0), mode=('IntPin', 0), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        img(cv2.flip(input.image, mode))        