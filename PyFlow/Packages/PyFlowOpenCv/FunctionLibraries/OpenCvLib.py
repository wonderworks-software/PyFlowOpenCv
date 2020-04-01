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


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape


class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        colors = OrderedDict({
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)})
        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, image, c):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)
            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                minDist = (d, i)
        # return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]


class OpenCvLib(FunctionLibraryBase):
    '''doc string for OpenCvLib'''

    def __init__(self, packageName):
        super(OpenCvLib, self).__init__(packageName)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Inputs', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_ReadImage(path=('StringPin', "",{PinSpecifires.INPUT_WIDGET_VARIANT: "FilePathWidget"}), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        image = cv2.imread(path)
        image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img(image)           

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Transformations', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_FlipImage(input=('ImagePin', 0), mode=('IntPin', 0), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        image = input.image
        img(cv2.flip(image, mode))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Edge Detection', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def auto_canny(input=('ImagePin', 0), sigma=('FloatPin', 0.33), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        image = input.image
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        img(cv2.Canny(image, lower, upper))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Process', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def shape_detector(input=('ImagePin', 0), thres1=('IntPin', 20), thres2=('IntPin', 255),
                       img=(REF, ('ImagePin', 0)), _thresh=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        image = input.image
        blurred = cv2.GaussianBlur(input.image, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        thresh = cv2.threshold(gray, thres1, thres2, cv2.THRESH_BINARY)[1]
        _thresh(thresh)
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # initialize the shape detector and color labeler
        sd = ShapeDetector()
        cl = ColorLabeler()

        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / (1 + M["m00"])))
            cY = int((M["m01"] / (1 + M["m00"])))
            # detect the shape of the contour and label the color
            shape = sd.detect(c)
            color = cl.label(lab, c)
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape and labeled
            # color on the image
            c = c.astype("int")
            text = "{} {}".format(color, shape)
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, text, (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        img(image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Color', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def color_filter(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)), _thresh=(REF, ('ImagePin', 0))):
        """filter a BGR color range from image"""
        image = input.image
        # define the list of boundaries
        boundaries = [  # would be good to have a BGR colour picker..
            ([17, 15, 100], [50, 56, 200]),  # RBG lower and upper limit for RED
            ([86, 31, 4], [220, 88, 50]),  # RBG lower and upper limit for GREEN
            ([25, 146, 190], [62, 174, 250]),  # RBG lower and upper limit for BLUE
            ([103, 86, 65], [145, 133, 128])  # RBG lower and upper limit for GREY (intensity)
        ]

        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask=mask)

        img(output)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Edge Detection', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def HED(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)), _canny=(REF, ('ImagePin', 0))):
        """Holistically-Nested Edge Detection"""
        image = input.image

        class CropLayer(object):
            def __init__(self, params, blobs):
                # initialize our starting and ending (x, y)-coordinates of
                # the crop
                self.startX = 0
                self.startY = 0
                self.endX = 0
                self.endY = 0

            def getMemoryShapes(self, inputs):
                # the crop layer will receive two inputs -- we need to crop
                # the first input blob to match the shape of the second one,
                # keeping the batch size and number of channels
                (inputShape, targetShape) = (inputs[0], inputs[1])
                (batchSize, numChannels) = (inputShape[0], inputShape[1])
                (H, W) = (targetShape[2], targetShape[3])
                # compute the starting and ending crop coordinates
                self.startX = int((inputShape[3] - targetShape[3]) / 2)
                self.startY = int((inputShape[2] - targetShape[2]) / 2)
                self.endX = self.startX + W
                self.endY = self.startY + H
                # return the shape of the volume (we'll perform the actual
                # crop during the forward pass
                return [[batchSize, numChannels, H, W]]

            def forward(self, inputs):
                # use the derived (x, y)-coordinates to perform the crop
                return [inputs[0][:, :, self.startY:self.endY,
                        self.startX:self.endX]]

        (H, W) = image.shape[:2]
        # convert the image to grayscale, blur it, and perform Canny edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blurred, 30, 150)

        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        # set the blob as the input to the network and perform a forward pass to compute the edges
        protoPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res", "deploy.prototxt")
        modelPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res", "hed_pretrained_bsds.caffemodel")
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        # register our new layer with the model
        cv2.dnn_registerLayer("Crop", CropLayer)
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")

        _canny(canny)
        img(hed)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Process', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def adapt_thres(input=('ImagePin', 0), _mean=(REF, ('ImagePin', 0)),
                    _thresh=(REF, ('ImagePin', 0)), _gaussian=(REF, ('ImagePin', 0))):
        """filter a BGR color range from image"""
        image = input.image

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        # here 11 is the pixel neighbourhood that is used to calculate the threshold value
        mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        images = [gray_image, thresh, mean, gaussian]

        _thresh(thresh)
        _mean(mean)
        _gaussian(gaussian)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Process', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def waters_hed(input=('ImagePin', 0), _thres=(REF, ('ImagePin', 0)),_sure_bg=(REF, ('ImagePin', 0)),
                    img=(REF, ('ImagePin', 0))):
        """filter a BGR color range from image"""
        image = input.image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # apply thresholding
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # get a kernel
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # extract the background from image
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_bg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        img(image)
        _thres(thresh)
        _sure_bg(sure_fg)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Process', NodeMeta.KEYWORDS: []})
    def bit_and(input=('ImagePin', 0),mask=('ImagePin', 0), img=(REF, ('ImagePin', 0)), _mask=(REF, ('ImagePin', 0))):
        """Takes an image and mask and applied logic and operation"""
        ret, mask = cv2.threshold(mask.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img(cv2.bitwise_and(input.image, input.image, mask=mask))
        _mask(mask)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Edge Detection', NodeMeta.KEYWORDS: []})
    def detectLines(input=('ImagePin', 0),
                    low_threshold=('IntPin', 50),
                    high_threshold=('IntPin', 150),
                    rho=('IntPin', 1),
                    theta=('FloatPin', np.pi/180),
                    threshold=('IntPin', 15),
                    min_line_length=('IntPin', 50),
                    max_line_gap=('IntPin', 20),

                    img=(REF, ('ImagePin', 0)), _mask=(REF, ('ImagePin', 0))):
        """Detects lines and crossings"""
        import CV_classes.poly_point_isect as bot

        gray = cv2.cvtColor(input.image,cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        # low_threshold = 50
        # high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        points = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        intersections = bot.isect_segments(points)

        for inter in intersections:
            a, b = inter
            for i in range(3):
                for j in range(3):
                    lines_edges[int(b) + i, int(a) + j] = [0, 255, 0]

        cv2.imwrite('line_parking.png', lines_edges)
