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

class ImageFilteringLib(FunctionLibraryBase):
    '''doc string for ImageFilteringLib'''

    def __init__(self, packageName):
        super(ImageFilteringLib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_BilateralFilter(input=('ImagePin', None), radius=('IntPin',9), 
                        sigmaColor=('FloatPin', 75), sigmaSpace=('FloatPin', 75), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.bilateralFilter(input, radius, sigmaColor, sigmaSpace)
        img(image)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Blur(input=('ImagePin', None), xradius=('IntPin',5), yradius=('IntPin',5), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.blur(input, (xradius,yradius))
        img(image)   

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_BoxFilter(input=('ImagePin', None), ddepth=('IntPin',-1), xradius=('IntPin',5), yradius=('IntPin',5),
                    normalize=('BoolPin',True), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.boxFilter(input, ddepth, (xradius,yradius), normalize =normalize)
        img(image)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Dilate(input=('ImagePin', None), kernel=('ImagePin',0), iterations =('IntPin',1),  img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.dilate(input, kernel, iterations=iterations)
        img(image)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Erode(input=('ImagePin', None), kernel=('ImagePin',0), iterations =('IntPin',1),  img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.erode(input, kernel, iterations=iterations)
        img(image)          

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Filter2D(input=('ImagePin', None), ddepth=('IntPin',-1), kernel=('ImagePin',0), delta =('FloatPin', 0),  img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.filter2D(input, ddepth,kernel, delta =delta )
        img(image)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_GetStructuringElement(shape=('IntPin',0), xsize=('IntPin',10), ysize=('IntPin',10), 
                                xcenter=('IntPin',10), ycenter=('IntPin',10),img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        shapes = [cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE]
        image = cv2.getStructuringElement(shapes[clamp(shape,0,2)], (xsize,ysize),(clamp(xcenter,0,xsize),clamp(ycenter,0,ysize)))
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_GetGaussianKernel(ksize=('IntPin',1), sigma=('FloatPin',-1), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.getGaussianKernel(oddify(ksize),sigma )
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_GetGaborKernel(xsize=('IntPin',2), ysize=('IntPin',2), 
                        sigma=('FloatPin',0), theta=('FloatPin',0),
                        lambd=('FloatPin',0), gamma=('FloatPin',0),
                        psi=('FloatPin',3.1416*0.5), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.getGaborKernel((xsize,ysize), sigma, theta, lambd, gamma, psi)
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_GetDerivKernels(dx=('IntPin',2), dy=('IntPin',2), 
                        ksize=('IntPin',0), normalize =('BoolPin',False),
                        kx=(REF, ('ImagePin', None)), ky=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        if ksize == 0:
            ksize = cv2.FILTER_SCHARR
        x,y = cv2.getDerivKernels(dx, dy, ksize, normalize)
        kx(x)
        ky(y)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_GaussianBlur(input=('ImagePin', None), xradius=('IntPin',5), yradius=('IntPin',5), 
                        sigmaX=('FloatPin',0), sigmaY=('FloatPin',0), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.GaussianBlur(input, (oddify(xradius),oddify(yradius)),sigmaX,sigmaY)
        img(image) 

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Laplacian(input=('ImagePin', None), ddepth=('IntPin',-1), 
                    ksize =('IntPin',1), scale =('FloatPin',1.0), 
    				delta =('FloatPin',0), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.Laplacian(input, ddepth=ddepth,ksize=oddify(ksize),scale=scale,delta=delta)
        img(image) 

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_MedianBlur(input=('ImagePin', None), radius=('IntPin',5), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.medianBlur(input, oddify(radius))
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_PyrDown(input=('ImagePin', None), xsize=('IntPin',120), ysize=('IntPin',120), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.pyrDown(input, (xsize,ysize))
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_PyrUp(input=('ImagePin', None), xsize=('IntPin',120), ysize=('IntPin',120), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.pyrUp(input, (xsize,ysize))
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_PyrMeanShiftFiltering(input=('ImagePin', None), sp=('FloatPin',1.5), sr=('FloatPin',1.5),
                                 maxLevel =('IntPin',1), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.pyrMeanShiftFiltering(input, sp, sr, maxLevel)
        img(image)        

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Scharr(input=('ImagePin', None),  ddepth=('IntPin',-1),
                    dx=('IntPin',3), dy=('IntPin',3),
                    scale =('FloatPin',1.0), delta  =('FloatPin',0), 
                    img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.Scharr(input, ddepth, dx, dy, scale, delta )
        img(image)   

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_SepFilter2D(input=('ImagePin', None),  ddepth=('IntPin',-1),
                    kernelX=('ImagePin',0), kernelY=('ImagePin',0), 
                    delta  =('FloatPin',0), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image = cv2.sepFilter2D(input, ddepth, kernelX, kernelY, delta = delta )
        img(image)        

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_Sobel(input=('ImagePin', None), ddepth=('IntPin',-1),
                 dx=('IntPin',3), dy=('IntPin',3),
                 ksize =('IntPin',3), scale =('FloatPin',1.0), 
                 delta  =('FloatPin',0), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        if ksize == 0:
            ksize = cv2.FILTER_SCHARR        
        image = cv2.Sobel(input, ddepth, dx, dy, ksize= ksize, scale=scale, delta=delta)
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_SpatialGradient(input=('ImagePin', None), ksize =('IntPin',3),
                 dx=(REF, ('ImagePin', None)), dy=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        x,y = cv2.spatialGradient(input, ksize)
        dx(x)
        dy(y)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: []})
    # Blurs An image
    def cv_SqrBoxFilter(input=('ImagePin', None), ddepth=('IntPin',-1),
                        xsize =('IntPin',3), ysize =('IntPin',3),
                        normalize=('BoolPin',True), img=(REF, ('ImagePin', None))):
        """ Blurs An image."""
        image= cv2.sqrBoxFilter(input, ddepth,(xsize,ysize),normalize =normalize )
        img(image)

 