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
    def OpticalFlowSparseToDense(previmg=('ImagePin', None), img=('ImagePin', None),
                                 grid_step=('IntPin',8),k=('IntPin',128),
                                 sigma = ('FloatPin',0.05),use_post_proc=('BoolPin',True),
                                 fgs_lambda = ('FloatPin',500),fgs_sigma =('FloatPin',1.5),
                                 flowVis=(REF, ('ImagePin', None)),flowVisCombined=(REF, ('ImagePin', None))):
        prevFrame = previmg
        frame = img
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
    def OpticalFlowSF(previmg=('ImagePin', None), img=('ImagePin', None),
                                 layers=('IntPin',3),averaging_block_size=('IntPin',2),
                                 max_flow=('IntPin',4),
                                 sigma_dist = ('FloatPin',4.1), sigma_color = ('FloatPin',25.5),
                                 postprocess_window =('IntPin',18),
                                 sigma_dist_fix = ('FloatPin',55.0), sigma_color_fix = ('FloatPin',25.5),
                                 occ_thr  = ('FloatPin',0.35),upscale_averaging_radius=('IntPin',18),
                                 upscale_sigma_dist  = ('FloatPin',55.0),upscale_sigma_color = ('FloatPin',25.5),
                                 speed_up_thr = ('FloatPin',10),
                                 flowVis=(REF, ('ImagePin', None)),flowVisCombined=(REF, ('ImagePin', None))):
        prevFrame = previmg
        frame = img
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

class LK_optical_flow_Lib(FunctionLibraryBase):
    '''doc string for OpenCvLib'''

    previous_image = None
    previous_points=None
    mask_image = None

    def __init__(self, packageName):
        super(LK_optical_flow_Lib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'OpticalFlow', NodeMeta.KEYWORDS: []})
    def LK_optical_flow(
            input=('ImagePin', None),
            previous_points=('KeyPointsPin', 0),
            img=(REF, ('ImagePin', None)) ):
        color = np.random.randint(0, 255, (100, 3))
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if LK_optical_flow_Lib.previous_image is None \
                or LK_optical_flow_Lib.previous_image.shape!=input.shape:
            LK_optical_flow_Lib.previous_image= input

        if LK_optical_flow_Lib.mask_image is None \
                or LK_optical_flow_Lib.mask_image.shape[:2]!=input.shape[:2]:
            LK_optical_flow_Lib.mask_image= np.zeros_like(input)
            LK_optical_flow_Lib.mask_image= cv2.cvtColor(LK_optical_flow_Lib.mask_image,cv2.COLOR_GRAY2BGR)

        LK_optical_flow_Lib.previous_points=previous_points.data
        color_draw= cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        prev_gray =LK_optical_flow_Lib.previous_image
        gray = input
        if previous_points:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, LK_optical_flow_Lib.previous_points, None, **lk_params)
            good_new = p1[st == 1]
            good_old = previous_points.data[st == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                LK_optical_flow_Lib.mask_image= cv2.line(LK_optical_flow_Lib.mask_image, (a, b), (c, d), color[i].tolist(), 2)
                # color_draw= cv2.line(color_draw, (a, b), (c, d), color[i].tolist(), 2)
                color_draw= cv2.circle(color_draw, (a, b), 5, color[i].tolist(), -1)
            color_draw= cv2.add(color_draw, LK_optical_flow_Lib.mask_image)
            LK_optical_flow_Lib.previous_points=good_new
        img(color_draw)
        LK_optical_flow_Lib.previous_image=input


class Dense_optical_flow_Lib(FunctionLibraryBase):
    '''doc string for OpenCvLib'''

    previous_image = None

    def __init__(self, packageName):
        super(Dense_optical_flow_Lib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'OpticalFlow', NodeMeta.KEYWORDS: []})
    def Dense_optical_flow(
            input=('ImagePin', None),
            img=(REF, ('ImagePin', None)) ):

        if Dense_optical_flow_Lib.previous_image is None \
                or Dense_optical_flow_Lib.previous_image.shape!=input.shape:
            Dense_optical_flow_Lib.previous_image= input
        # Sets image saturation to maximum
        color_img= cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        mask= np.zeros_like(color_img)
        mask[..., 1] = 255

        prev_gray =Dense_optical_flow_Lib.previous_image
        gray = input
        if gray is not None and prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray , None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                                poly_n=5, poly_sigma=1.1, flags=0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            dense_flow = cv2.addWeighted(color_img, 1, rgb, 2, 0)
            img(dense_flow)
        Dense_optical_flow_Lib.previous_image=input