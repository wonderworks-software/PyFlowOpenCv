import cv2
import numpy as np

from PyFlow.Core import (
    FunctionLibraryBase,
    IMPLEMENT_NODE
)
from PyFlow.Core.Common import *
from PyFlow.Packages.PyFlowOpenCv.CV_classes.blend_modes import *
from PyFlow.Packages.PyFlowOpenCv.CV_classes.imageUtils import *


class ImageBlendingLib(FunctionLibraryBase):
    '''doc string for ImageBlendingLib'''

    def __init__(self, packageName):
        super(ImageBlendingLib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Blending', NodeMeta.KEYWORDS: []})
    def cv_AlphaMixImages(FG=('ImagePin', None),BG=('ImagePin', None),Mask=('ImagePin', None), img=(REF, ('ImagePin', None))):
        foreground = FG.astype(float)
        background = BG.astype(float)
        alpha = Mask.astype(float)/255
        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)
        # Add the masked foreground and background.
        outImage = cv2.normalize(cv2.add(foreground, background), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img(outImage)  

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Blending', NodeMeta.KEYWORDS: []})
    def bit_and(input=('ImagePin', None), mask=('ImagePin', None), img=(REF, ('ImagePin', None)), _mask=(REF, ('ImagePin', None))):
        """Takes an image and mask and applied logic and operation"""
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img(cv2.bitwise_and(input, input, mask=mask))
        _mask(mask)        

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Blending', NodeMeta.KEYWORDS: []})
    def cv_BlendImages( base_image=('ImagePin', None),overlay_image=('ImagePin', None), Mask=('ImagePin', None),
                        blendmode=('StringPin', 0,
                            {PinSpecifiers.VALUE_LIST: ['NORMAL', 'MULTIPLY', 'DARKEN', 'LIGHTEN', 'ADD', 
                            'COLOR_BURN', 'COLOR_DODGE', 'REFLECT', 'GLOW', 'OVERLAY', 'DIFFERENCE', 'NEGATION', 'SCREEN', 'XOR',
                            'SUBTRACT', 'DIVIDE', 'EXCLUSION', 'SOFT_LIGHT']}),
                        img=(REF, ('ImagePin', None)), center=('BoolPin', True,),
                        interpolation=('StringPin', "INTER_CUBIC",
                            {PinSpecifiers.VALUE_LIST: ["INTER_LINEAR","INTER_CUBIC","INTER_AREA","INTER_LANCZOS4",
                            "INTER_LINEAR_EXACT","INTER_MAX","WARP_FILL_OUTLIERS","WARP_INVERSE_MAP" ]} )
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

        modes = {key:value for key, value in BlendModes.__dict__.items() if not key.startswith('__') and not callable(key)}        
        blend_mode = modes[blendmode]
        base = normalize_image(base_image)
        overlay = normalize_image(overlay_image)

        b_h, b_w, b_c = get_h_w_c(base)
        o_h, o_w, _ = get_h_w_c(overlay)
        m_h, m_w, m_c = get_h_w_c(Mask)

        max_h = max(b_h, o_h)
        max_w = max(b_w, o_w)

        if (b_w, b_h) == (o_w, o_h):
            # we don't have to do any size adjustments
            result = blend_images(overlay, base, blend_mode)
            result_c = b_c
        else:
            overlay = resize_to_fit(overlay,base,inter[interpolation])
            overlay = expand_image_to_fit(overlay, base, center)
            result = blend_images(
                overlay,
                base,
                blend_mode,
            )
            result_c = get_h_w_c(result)[2]

        if Mask.any():
            alpha = normalize_image(Mask)
            if (m_h, m_w) != (b_h, b_w):
                alpha = resize_to_fit(alpha, base, inter[interpolation])
                alpha = expand_image_to_fit(alpha, base, center)
            alpha_c = get_h_w_c(alpha)[2]

            if result_c == 4 and b_c != 4:
                base = convert_to_BGRA(base, b_c)
            if result_c == 4 and alpha_c != 4:
                alpha = convert_to_BGRA(alpha, alpha_c)
            result = cv2.multiply(alpha, result)
            background = cv2.multiply(1.0 - alpha, base)
            result = cv2.add(result,background)
        img(cv2.normalize(np.clip(result, 0, 1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

