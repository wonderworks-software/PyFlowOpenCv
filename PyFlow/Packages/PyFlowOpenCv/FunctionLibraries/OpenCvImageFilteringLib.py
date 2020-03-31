import os
from collections import OrderedDict

import imutils as imutils
import numpy as np
from scipy.spatial import distance as dist
import cv2

from PyFlow.Core import (
    FunctionLibraryBase,
    IMPLEMENT_NODE
)
from PyFlow.Core.Common import *

class OpenCvImageFilteringLib(FunctionLibraryBase):
    '''doc string for OpenCvImageFilteringLib'''

    def __init__(self, packageName):
        super(OpenCvImageFilteringLib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_BilateralFilter(input=('ImagePin', 0), radius=('IntPin',9), 
                        sigmaColor=('FloatPin', 75), sigmaSpace=('FloatPin', 75), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.bilateralFilter(input.image, radius, sigmaColor, sigmaSpace)
        img(image)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_Blur(input=('ImagePin', 0), xradius=('IntPin',5), yradius=('IntPin',5), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.blur(input.image, (xradius,yradius))
        img(image)   

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_BoxFilter(input=('ImagePin', 0), ddepth=('IntPin',-1), xradius=('IntPin',5), yradius=('IntPin',5),
                    normalize=('BoolPin',True), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.boxFilter(input.image, ddepth, (xradius,yradius), normalize =normalize)
        img(image)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_Dilate(input=('ImagePin', 0), kernel=('ImagePin',0), iterations =('IntPin',1),  img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.dilate(input.image, kernel.image, iterations=iterations)
        img(image)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_Erode(input=('ImagePin', 0), kernel=('ImagePin',0), iterations =('IntPin',1),  img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.erode(input.image, kernel.image, iterations=iterations)
        img(image)          

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_Filter2D(input=('ImagePin', 0), ddepth=('IntPin',-1), kernel=('ImagePin',0), delta =('FloatPin', 0),  img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.filter2D(input.image, ddepth,kernel.image, delta =delta )
        img(image)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_GetStructuringElement(shape=('IntPin',0), xsize=('IntPin',10), ysize=('IntPin',10), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        shapes = [cv.MORPH_RECT,cv.MORPH_CROSS,cv.MORPH_ELLIPSE]
        image = cv2.getStructuringElement(shapes[min(max(0,shape),2)], (xsize,ysize))
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_GetGaussianKernel(ksize=('IntPin',1), sigma=('FloatPin',-1), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.getGaussianKernel(max(0,ksize+(ksize-1)),sigma )
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_GetGaborKernel(xsize=('IntPin',2), ysize=('IntPin',2), 
                        sigma=('FloatPin',0), theta=('FloatPin',0),
                        lambd=('FloatPin',0), gamma=('FloatPin',0),
                        psi=('FloatPin',3.1416*0.5), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.getGaborKernel((xsize,ysize), sigma, theta, lambd, gamma, psi)
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_GetDerivKernels(dx=('IntPin',2), dy=('IntPin',2), 
                        ksize=('IntPin',0), normalize =('BoolPin',False),
                        kx=(REF, ('ImagePin', 0)), ky=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        if ksize == 0:
            ksize = cv2.FILTER_SCHARR
        x,y = cv2.getDerivKernels(dx, dy, ksize, normalize)
        kx(x)
        ky(y)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_GaussianBlur(input=('ImagePin', 0), Xradius=('IntPin',5), Yradius=('IntPin',5), 
                        sigmaX=('IntPin',0), sigmaY=('IntPin',0), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.GaussianBlur(input.image, (max(0,xradius+(xradius-1)),max(0,yradius+(yradius-1))),sigmaX,sigmaY)
        img(image) 

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_Laplacian(input=('ImagePin', 0), ddepth=('IntPin',-1), 
                    ksize =('IntPin',1), scale =('FloatPin',1.0), 
    				delta =('FloatPin',0), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.Laplacian(input.image, ddepth,ksize,scale,delta)
        img(image) 

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_MedianBlur(input=('ImagePin', 0), radius=('IntPin',5), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.medianBlur(input.image, max(0,radius+(radius-1)))
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_PyrDown(input=('ImagePin', 0), xsize=('IntPin',120), ysize=('IntPin',120), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.pyrDown(input.image, (xsize,ysize))
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_PyrUp(input=('ImagePin', 0), xsize=('IntPin',120), ysize=('IntPin',120), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.pyrUp(input.image, (xsize,ysize))
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_PyrMeanShiftFiltering(input=('ImagePin', 0), sp=('FloatPin',120), sr=('FloatPin',120),
                                 maxLevel =('IntPin',1), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.pyrMeanShiftFiltering(input.image, sp, sr, maxLevel)
        img(image)        

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_Scharr(input=('ImagePin', 0),  ddepth=('IntPin',-1),
                    dx=('IntPin',3), dy=('IntPin',3),
                    scale =('FloatPin',1.0), delta  =('FloatPin',0), 
                    img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.Scharr(input.image, ddepth, dx, dy, scale, delta )
        img(image)   

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_SepFilter2D(input=('ImagePin', 0),  ddepth=('IntPin',-1),
                    kernelX=('ImagePin',0), kernelY=('ImagePin',0), 
                    delta  =('FloatPin',0), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image = cv2.sepFilter2D(input.image, ddepth, kernelX, kernelY, delta = delta )
        img(image)        

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_Sobel(input=('ImagePin', 0), ddepth=('IntPin',-1),
                 dx=('IntPin',3), dy=('IntPin',3),
                 ksize =('IntPin',3), scale =('FloatPin',1.0), 
                 delta  =('FloatPin',0), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        if ksize == 0:
            ksize = cv2.FILTER_SCHARR        
        image = cv2.Sobel(input.image, ddepth, dx, dy, ksize, scale, delta)
        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_SpatialGradient(input=('ImagePin', 0), ksize =('IntPin',3),
                 dx=(REF, ('ImagePin', 0)), dy=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        x,y = cv2.spatialGradient(input.image, ksize)
        dx(x)
        dy(y)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'ImageFiltering', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Blurs An image
    def cv_SqrBoxFilter(input=('ImagePin', 0), ddepth=('IntPin',-1),
                        xsize =('IntPin',3), ysize =('IntPin',3),
                        normalize=('BoolPin',True), img=(REF, ('ImagePin', 0))):
        """ Blurs An image."""
        image= cv2.sqrBoxFilter(input.image, ddepth,(xsize,ysize),normalize =normalize )
        img(image)

 