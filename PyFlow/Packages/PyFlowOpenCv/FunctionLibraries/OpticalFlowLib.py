import os
from collections import OrderedDict

import imutils as imutils
import numpy as np
from scipy.spatial import distance as dist
import cv2
from imutils.object_detection import non_max_suppression

from PyFlow.Core import (
    FunctionLibraryBase,
    IMPLEMENT_NODE
)
from PyFlow.Core.Common import *
import pytesseract


class OpticalFlowLib(FunctionLibraryBase):
    '''doc string for OpticalFlowLib'''

    def __init__(self, packageName):
        super(OpticalFlowLib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'OpticalFlow', NodeMeta.KEYWORDS: []})
    def OpticalFlowSparseToDense(previmg=('ImagePin', 0), img=('ImagePin', 0),
                                 grid_step=('IntPin',8),k=('IntPin',128),
                                 sigma = ('FloatPin',0.05),use_post_proc=('BoolPin',True),
                                 fgs_lambda = ('FloatPin',500),fgs_sigma =('FloatPin',1.5),
                                 flowVis=(REF, ('ImagePin', 0)),flowVisCombined=(REF, ('ImagePin', 0))):
        prevFrame = previmg.image
        frame = img.image
        hsv = np.zeros_like(prevFrame)
        hsv[..., 1] = 255
        prevGrey = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        flow = cv2.optflow.calcOpticalFlowSparseToDense(prevGrey, grey, None, grid_step,k,
                                                                            sigma, use_post_proc,
                                                                            fgs_lambda,fgs_sigma)
        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert hsv image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        dense_flow = cv2.addWeighted(frame, 1, bgr, 2, 0)     
        flowVis(bgr)
        flowVisCombined(dense_flow)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'OpticalFlow', NodeMeta.KEYWORDS: []})
    def OpticalFlowSF(previmg=('ImagePin', 0), img=('ImagePin', 0),
                                 layers=('IntPin',3),averaging_block_size=('IntPin',2),
                                 max_flow=('IntPin',4),
                                 sigma_dist = ('FloatPin',4.1), sigma_color = ('FloatPin',25.5),
                                 postprocess_window =('IntPin',18),
                                 sigma_dist_fix = ('FloatPin',55.0), sigma_color_fix = ('FloatPin',25.5),
                                 occ_thr  = ('FloatPin',0.35),upscale_averaging_radius=('IntPin',18),
                                 upscale_sigma_dist  = ('FloatPin',55.0),upscale_sigma_color = ('FloatPin',25.5),
                                 speed_up_thr = ('FloatPin',10),
                                 flowVis=(REF, ('ImagePin', 0)),flowVisCombined=(REF, ('ImagePin', 0))):
        prevFrame = previmg.image
        frame = img.image
        hsv = np.zeros_like(prevFrame)
        hsv[..., 1] = 255    
        flow = cv2.optflow.calcOpticalFlowSF(prevFrame, frame, layers,averaging_block_size,
                                                                    max_flow,sigma_dist,sigma_color,
                                                                    postprocess_window,sigma_dist_fix,
                                                                    sigma_color_fix,occ_thr,
                                                                    upscale_averaging_radius,upscale_sigma_dist,
                                                                    upscale_sigma_color,speed_up_thr)
        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert hsv image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        dense_flow = cv2.addWeighted(frame, 1, bgr, 2, 0)     
        flowVis(bgr)
        flowVisCombined(dense_flow)

