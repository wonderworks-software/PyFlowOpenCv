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
    def cv_ReadImage(path=('StringPin', "", {PinSpecifires.INPUT_WIDGET_VARIANT: "FilePathWidget"}),
                     gray_scale= ( 'BoolPin', False), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        if gray_scale:
            img(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        else:
            img(cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED))


    @staticmethod
    @IMPLEMENT_NODE(returns=None, nodeType=NodeTypes.Callable,
                    meta={NodeMeta.CATEGORY: 'Inputs', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_WriteImage(path=('StringPin', "", {PinSpecifires.INPUT_WIDGET_VARIANT: "FilePathWidget"}),
                      img=('ImagePin', 0)):
        """Return a frame of the loaded image."""
        cv2.imwrite(path, img.image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Inputs', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_ReadVideo(path=('StringPin', "", {PinSpecifires.INPUT_WIDGET_VARIANT: "FilePathWidget"}),
                     video=(REF, ('VideoPin', 0))):
        """Return a frame of the loaded image."""
        video(cv2.VideoCapture(path))
        
    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Inputs', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_WebCam(cam_index=('IntPin', 0), video=(REF, ('VideoPin', 0))):
        """Return a frame of the loaded image."""
        video(cv2.VideoCapture(cam_index))

    @staticmethod
    @IMPLEMENT_NODE(returns=('BoolPin', False),
                    meta={NodeMeta.CATEGORY: 'Inputs', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Return a random frame of x,y
    def cv_ReadNextFrame(video=('VideoPin', ""), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        ret, frame = video.read()
        if not ret:
            video.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        img(frame)
        return ret

    @staticmethod
    @IMPLEMENT_NODE(returns=('BoolPin', False),
                    meta={NodeMeta.CATEGORY: 'Inputs', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Return a random frame of x,y
    def cv_ReadNextTwoFrame(video=('VideoPin', ""),
                            img=(REF, ('ImagePin', 0)),
                            next_img=(REF, ('ImagePin', 0))
                            ):
        """Return a frame of the loaded image."""
        ret, frame = video.read()
        ret, next_frame = video.read()
        img(frame)
        next_img(next_frame)
        return ret

    @staticmethod
    @IMPLEMENT_NODE(returns=('BoolPin', False), meta={NodeMeta.CATEGORY: 'Inputs', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_ReadVideoFrame(video=('VideoPin', 0,), frame=('IntPin', 0), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        video.set(1, frame)
        ret, im = video.read()
        img(im)
        return ret

    @staticmethod
    @IMPLEMENT_NODE(returns=('BoolPin', False),
                    meta={NodeMeta.CATEGORY: 'Inputs', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
    # Return a random frame of x,y
    def cv_VideoisOpened(video=('VideoPin', 0)):
        """Return a frame of the loaded image."""
        return video.isOpened()

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Converters', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_BGR2RGB(input=('ImagePin', 0), img=(REF, ('ImagePin', 0))):
        img(cv2.cvtColor(input.image, cv2.COLOR_BGR2RGB))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Converters', NodeMeta.KEYWORDS: []})
    def cv_BGR2GRAY(input=('ImagePin', 0), img=(REF, ('ImagePin', 0))):
        img(cv2.cvtColor(input.image, cv2.COLOR_BGR2GRAY))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'OpenCl', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_convertToGPU(img=('ImagePin', 0), gpuImg=(REF, ('ImagePin', 0))):
        gpuImg(cv2.UMat(img.image))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'OpenCl', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_convertToCPU(gpuImg=('ImagePin', 0), img=(REF, ('ImagePin', 0))):
        if gpuImg.image.__class__.__name__ == "UMat":
            img(cv2.UMat.get(gpuImg.image))
        else:
            img(gpuImg)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Color', NodeMeta.KEYWORDS: []})
    def cv_HDR_AutoGamma(input=('ImagePin', 0), img=(REF, ('ImagePin', 0))):
        image = np.clip(np.power(input.image, 1.0 / 2.2), 0, 1)
        img(np.uint8(image * 255))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Color', NodeMeta.KEYWORDS: []})
    def cv_ConvertScaleAbs(input=('ImagePin', 0), img=(REF, ('ImagePin', 0))):
        img(cv2.convertScaleAbs(input.image))

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Transformations', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_FlipImage(input=('ImagePin', 0), mode=('IntPin', 0), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        img(cv2.flip(input.image, mode))

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
        modelPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                 "hed_pretrained_bsds.caffemodel")
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
    def waters_hed(input=('ImagePin', 0), _thres=(REF, ('ImagePin', 0)), _sure_bg=(REF, ('ImagePin', 0)),
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
    def bit_and(input=('ImagePin', 0), mask=('ImagePin', 0), img=(REF, ('ImagePin', 0)), _mask=(REF, ('ImagePin', 0))):
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
                    theta=('FloatPin', np.pi / 180),
                    threshold=('IntPin', 15),
                    min_line_length=('IntPin', 50),
                    max_line_gap=('IntPin', 20),

                    img=(REF, ('ImagePin', 0)), _mask=(REF, ('ImagePin', 0))):
        """Detects lines and crossings"""
        import CV_classes.poly_point_isect as bot

        gray = cv2.cvtColor(input.image, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

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

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Detection', NodeMeta.KEYWORDS: []})
    def classifcation_dnn(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                           keywords=(REF, ('GraphElementPin', 0)),
                           ):
        """Takes an image and mask and applied logic and operation"""

        face_model_proto_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                             "bvlc_googlenet.prototxt")
        face_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                       "bvlc_googlenet.caffemodel")
        keywords_path= os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                       "synset_words.txt")

        words_rows = open(keywords_path).read().strip().split("\n")
        classes = [r[r.find(" ") + 1:].split(",")[0] for r in words_rows]


        image = cv2.resize(input.image, (224, 224))
        image=image[:,:,:3]
        (h, w) = image.shape[:2]
        net = cv2.dnn.readNetFromCaffe(face_model_proto_path,face_model_path)
        blob = cv2.dnn.blobFromImage(image, 1.0, (w,h),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        net.setInput(blob)
        preds= net.forward()
        idxs = np.argsort(preds[0])[::-1][:5]

        # loop over the top-5 predictions and display them
        words=[]
        for (i, idx) in enumerate(idxs):
            # draw the top prediction on the input image
            if i == 0:
                text = "Label: {}, {:.2f}%".format(classes[idx],
                                                   preds[0][idx] * 100)
                words.append(text)
                break
        keywords({'text':words})
        img(input.image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Detection', NodeMeta.KEYWORDS: []})
    def yolo_dnn(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                 detections=(REF, ('GraphElementPin', 0)),
                 ):
        """Takes an image and mask and applied logic and operation"""

        yolo_model_proto_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                             "yolov3-tiny.weights")
        yolo_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                       "yolov3-tiny.cfg")
        keywords_path= os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                    "yolov3.txt")

        modelSize = 416

        with open(keywords_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        (image_height, image_width) = input.image.shape[:2]

        image = cv2.resize(input.image, (modelSize, modelSize))
        image=image[:,:,:3]
        (h, w) = image.shape[:2]
        rW = w/ float(modelSize)
        rH = h/ float(modelSize)
        scale = 0.00392
        net= cv2.dnn.readNet(yolo_model_proto_path,yolo_model_path)
        blob = cv2.dnn.blobFromImage(image, scale, (w,h),
                                     mean=(0,0,0),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        retval= net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in retval:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    if detection[2]>1 or detection[3]>1:
                        continue
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)

                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    box=[x, y, w, h]
                    boxes.append(box)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        annotation=[]
        if len(indices) != 0:

            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                annotation.append({'box': (x, y, w, h), 'class': classes[class_ids[i]], 'confidence': confidences[i]})
        detections({'detection': annotation})
        img(input.image)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Detection', NodeMeta.KEYWORDS: []})
    def face_detection(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                       rects=(REF, ('GraphElementPin', 0)),
                       scaleFactor=('FloatPin', 1.1),
                       minNeighbores=('IntPin', 4)
                       ):
        """Takes an image and mask and applied logic and operation"""

        face_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                       "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(face_model_path)
        gray = cv2.cvtColor(input.image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbores)
        img(gray)
        rects({'rect': [(x, y, w, h) for (x, y, w, h) in faces]})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Detection', NodeMeta.KEYWORDS: []})
    def face_detection_dnn(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                       rects=(REF, ('GraphElementPin', 0)),
                       ):
        """Takes an image and mask and applied logic and operation"""

        face_model_proto_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                       "face_detect_deploy.prototxt")
        face_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                       "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        (h, w) = input.image.shape[:2]
        net = cv2.dnn.readNetFromCaffe(face_model_proto_path,face_model_path)
        blob = cv2.dnn.blobFromImage(cv2.resize(input.image, (300, 300)), 1.0, (300, 300),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        boxs=[]
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxs.append((startX,startY,endX-startX,endY-startY))
        img(input.image)

        rects({'rect': boxs})
    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Detection', NodeMeta.KEYWORDS: []})
    def text_detection_dnn(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                           rects=(REF, ('GraphElementPin', 0)),
                           ):
        """Takes an image and mask and applied logic and operation"""

        text_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                       "frozen_east_text_detection.pb")
        modelSize=320
        (h, w) = input.image.shape[:2]
        image = cv2.resize(input.image, (modelSize,modelSize))
        rW = w/ float(modelSize)
        rH = h/ float(modelSize)
        net = cv2.dnn.readNet(text_model_path)
        blob = cv2.dnn.blobFromImage(image, 1.0, (modelSize,modelSize),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        (scores, geometry) = net.forward(layerNames)
        # scores = net.forward()
        (numRows, numCols) = scores.shape[2:4]
        boxes= []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.5:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                # box = (x * rW, y * rH, w * rW, h*rH)
                boxes.append((startX*rW,startY*rH,(endX)*rW,(endY)*rH))
                confidences.append(scoresData[x])
        filtered_boxes = non_max_suppression(np.array(boxes), probs=confidences)
        boxes=[(startX,startY,endX-startX,endY-startY) for startX,startY,endX,endY in filtered_boxes]
        img(input.image)
        rects({'rect': boxes})


    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Detection', NodeMeta.KEYWORDS: []})
    def eye_detection(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                      rects=(REF, ('GraphElementPin', 0)),
                      scaleFactor=('FloatPin', 1.1),
                      minNeighbores=('IntPin', 4)
                      ):
        """Takes an image and mask and applied logic and operation"""

        eye_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res", "haarcascade_eye.xml")
        eye_cascade = cv2.CascadeClassifier(eye_model_path)
        gray = cv2.cvtColor(input.image, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor, minNeighbores)
        img(gray)
        rects({'rect': [(x, y, w, h) for (x, y, w, h) in eyes]})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def goodFeatureToTrack(input=('ImagePin', 0), keypoints=(REF, ('KeyPointsPin', 0)),
                           draw_points=(REF, ('GraphElementPin', 0))):
        """Takes an image and mask and applied logic and operation"""
        gray = cv2.cvtColor(input.image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        if corners is not None:
            # corners = np.float32(corners)
            keypoints(corners)
            draw_points({'point': [(item[0][0], item[0][1]) for item in corners]})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def SURF_Feature(input=('ImagePin', 0),
                     keypoints=(REF, ('KeyPointsPin', 0)),
                     descriptor=(REF, ('DescriptorPin', 0)),
                     draw_key_points=(REF, ('GraphElementPin', 0)),
                     draw_points=(REF, ('GraphElementPin', 0))):
        """Takes an image and mask and applied logic and operation"""

        surf = cv2.xfeatures2d.SURF_create(400)
        kp, des = surf.detectAndCompute(input.image, None)
        if kp and len(kp):
            # corners = np.float32(corners)
            keypoints((kp,))
            descriptor(des)
            draw_points({'point': [(item.pt[0], item.pt[1]) for item in kp]})
            draw_key_points({'key_point': kp})


    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def SIFT_Feature(input=('ImagePin', 0),
                     keypoints=(REF, ('KeyPointsPin', 0)),
                     descriptor=(REF, ('DescriptorPin', 0)),
                     draw_key_points=(REF, ('GraphElementPin', 0)),
                     draw_points=(REF, ('GraphElementPin', 0))):
        """Takes an image and mask and applied logic and operation"""

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(input.image, None)
        if kp and len(kp):
            # corners = np.float32(corners)
            keypoints((kp,))
            descriptor(des)
            draw_points({'point': [(item.pt[0], item.pt[1]) for item in kp]})
            draw_key_points({'key_point': kp})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def FAST_Feature(input=('ImagePin', 0),
                     keypoints=(REF, ('KeyPointsPin', 0)),
                     descriptor=(REF, ('DescriptorPin', 0)),
                     draw_key_points=(REF, ('GraphElementPin', 0)),
                     draw_points=(REF, ('GraphElementPin', 0))):
        """Takes an image and mask and applied logic and operation"""

        fast= cv2.FastFeatureDetector_create()
        kp, des = fast.detectAndCompute(input.image, None)
        if kp and len(kp):
            # corners = np.float32(corners)
            keypoints((kp,))
            descriptor(des)
            draw_points({'point': [(item.pt[0], item.pt[1]) for item in kp]})
            draw_key_points({'key_point': kp})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def BRISK_Feature(input=('ImagePin', 0),
                      keypoints=(REF, ('KeyPointsPin', 0)),
                      descriptor=(REF, ('DescriptorPin', 0)),
                      draw_key_points=(REF, ('GraphElementPin', 0)),
                      draw_points=(REF, ('GraphElementPin', 0))):
        """Takes an image and mask and applied logic and operation"""

        brisk= cv2.BRISK_create()
        kp, des = brisk.detectAndCompute(input.image, None)
        if kp and len(kp):
            # corners = np.float32(corners)
            keypoints((kp,))
            descriptor(des)
            draw_points({'point': [(item.pt[0], item.pt[1]) for item in kp]})
            draw_key_points({'key_point': kp})


    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def AKAZE_Feature(input=('ImagePin', 0),
                     keypoints=(REF, ('KeyPointsPin', 0)),
                     descriptor=(REF, ('DescriptorPin', 0)),
                     draw_key_points=(REF, ('GraphElementPin', 0)),
                     draw_points=(REF, ('GraphElementPin', 0))):
        """Takes an image and mask and applied logic and operation"""

        akaze= cv2.AKAZE_create()
        kp, des = akaze.detectAndCompute(input.image, None)
        if kp and len(kp):
            # corners = np.float32(corners)
            keypoints((kp,))
            descriptor(des)
            draw_points({'point': [(item.pt[0], item.pt[1]) for item in kp]})
            draw_key_points({'key_point': kp})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def BRIEF_Feature(input=('ImagePin', 0),
                     keypoints=(REF, ('KeyPointsPin', 0)),
                     descriptor=(REF, ('DescriptorPin', 0)),
                     draw_key_points=(REF, ('GraphElementPin', 0)),
                     draw_points=(REF, ('GraphElementPin', 0))):
        """Takes an image and mask and applied logic and operation"""

        star = cv2.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        # find the keypoints with STAR
        kp = star.detect(input.image, None)
        # compute the descriptors with BRIEF
        kp, des = brief.compute(input.image, kp)
        if kp and len(kp):
            # corners = np.float32(corners)
            keypoints((kp,))
            descriptor(des)
            draw_points({'point': [(item.pt[0], item.pt[1]) for item in kp]})
            draw_key_points({'key_point': kp})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def ORB_Feature_Detector(input=('ImagePin', 0), keypoints=(REF, ('KeyPointsPin', 0)),
                             draw_key_points=(REF, ('GraphElementPin', 0)),
                             draw_points=(REF, ('GraphElementPin', 0))):
        """Takes an image and mask and applied logic and operation"""
        orb = cv2.ORB_create(nfeatures=2000)
        kp = orb.detect(input.image)
        if kp and len(kp):
            # corners = np.float32(corners)
            keypoints((kp,))
            draw_points({'point': [(item.pt[0], item.pt[1]) for item in kp]})
            draw_key_points({'key_point': kp})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def ORB_Feature_Extraction(input=('ImagePin', 0), keypoints=('KeyPointsPin', 0),
                               descriptor=(REF, ('DescriptorPin',0))
                               ):
        """Takes an image and mask and applied logic and operation"""
        orb = cv2.ORB_create(nfeatures=2000)
        kp, des = orb.compute(input.image, keypoints.data[0])
        descriptor(des)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'VideoAnalysis', NodeMeta.KEYWORDS: []})
    def CreateBackgroundSubtractorMOG2(
            history=('IntPin', 400),
            threshold=('FloatPin', 16),
            detectShadow=('BoolPin', True),
            background_subtrator=(REF, ('BackgroundSubtractorPin', 0))):
        backSub = cv2.createBackgroundSubtractorMOG2(history,threshold,detectShadow)
        background_subtrator(backSub)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'VideoAnalysis', NodeMeta.KEYWORDS: []})
    def CreateBackgroundSubtractorKNN(
            history=('IntPin', 400),
            threshold=('FloatPin', 16),
            detectShadow=('BoolPin', True),
            background_subtrator=(REF, ('BackgroundSubtractorPin', 0))):
        backSub = cv2.createBackgroundSubtractorKNN(history,threshold,detectShadow)
        background_subtrator(backSub)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'VideoAnalysis', NodeMeta.KEYWORDS: []})
    def BackgroundSubtract(input=('ImagePin', 0), background_subtrator=('BackgroundSubtractorPin', 0),
                           fgMask=(REF, ('ImagePin', 0))):
        mask = background_subtrator.data.apply(input.image)
        fgMask(mask)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def KNN_Match(
            descriptor_1=('DescriptorPin',0),
            descriptor_2=('DescriptorPin', 0),
            good_ratio=('FloatPin',0.75),
            match=(REF, ('FeatureMatchPin', 0)) ):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptor_1.data, descriptor_2.data, 2)
        good = []
        for m, n in matches:
            if m.distance < good_ratio * n.distance:
                good.append([m])
        match(good)


    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def FLANN_Match(
            descriptor_1=('DescriptorPin',0),
            descriptor_2=('DescriptorPin', 0),
            good_ratio=('FloatPin',0.75),
            match=(REF, ('FeatureMatchPin', 0)) ):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptor_1.data, descriptor_2.data, 2)
        good = []
        for m, n in matches:
            if m.distance < good_ratio * n.distance:
                good.append([m])
        match(good)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def Draw_Match(
            input_1=('ImagePin', 0), keypoints_1=('KeyPointsPin', 0),
            input_2=('ImagePin', 0), keypoints_2=('KeyPointsPin', 0),
            matches= ('FeatureMatchPin',0),
            output=(REF, ('ImagePin', 0)) ):
        """Takes an image and mask and applied logic and operation"""
        img3 = cv2.drawMatchesKnn(input_1.image, keypoints_1.data[0], input_2.image, keypoints_2.data[0], matches,
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        output(img3)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Feature', NodeMeta.KEYWORDS: []})
    def Dense_optical_flow(
            prev_image=('ImagePin', 0),
            image=('ImagePin', 0),
            img=(REF, ('ImagePin', 0)) ,
            output=(REF, ('ImagePin', 0)) ):

        # Create mask
        # mask = np.zeros(prev_image.image.shape+(3,),dtype=np.float32)
        mask = np.zeros_like(prev_image.image)
        # Sets image saturation to maximum
        mask[..., 1] = 255
        img(image)
        prev_gray = cv2.cvtColor(prev_image.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray , None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                            poly_n=5, poly_sigma=1.1, flags=0)
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        # Open a new window and displays the output frame
        dense_flow = cv2.addWeighted(image.image, 1, rgb, 2, 0)
        output(dense_flow)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Histogram', NodeMeta.KEYWORDS: []})
    def cv_Histogram(input=('ImagePin', 0), img=(REF, ('ImagePin', 0))):

        bgr_planes = cv2.split(input.image)
        histSize = 256
        histRange = (0, 256)  # the upper boundary is exclusive
        accumulate = False
        b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
        hist_w = 512
        hist_h = 400
        bin_w = int(round(hist_w / histSize))
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        for i in range(1, histSize):
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(round(b_hist[i - 1][0]))),
                    (bin_w * (i), hist_h - int(round(b_hist[i][0]))),
                    (255, 0, 0), thickness=2)
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(round(g_hist[i - 1][0]))),
                    (bin_w * (i), hist_h - int(round(g_hist[i][0]))),
                    (0, 255, 0), thickness=2)
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(round(r_hist[i - 1][0]))),
                    (bin_w * (i), hist_h - int(round(r_hist[i][0]))),
                    (0, 0, 255), thickness=2)

        img(histImage)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Process', NodeMeta.KEYWORDS: []})
    # Return a random frame of x,y
    def cv_Threahold(input=('ImagePin', 0), threshold=('IntPin', 127), img=(REF, ('ImagePin', 0))):
        """Return a frame of the loaded image."""
        threshold, cv__threshold = cv2.threshold(input.image, threshold, 255, cv2.THRESH_BINARY)
        img(cv__threshold)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Detection', NodeMeta.KEYWORDS: []})
    def gender_dnn(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                          keywords=(REF, ('GraphElementPin', 0)),
                          ):
        """Takes an image and mask and applied logic and operation"""

        gender_proto_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                             "deploy_gender2.prototxt")
        gender_caffe_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "res",
                                       "gender_net.caffemodel")

        image = cv2.resize(input.image, (227, 227))
        image=image[:,:,:3]
        (h, w) = image.shape[:2]
        net = cv2.dnn.readNetFromCaffe(gender_proto_path,gender_caffe_path)
        blob = cv2.dnn.blobFromImage(image, 1.0, (w,h),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        net.setInput(blob)
        preds= net.forward()
        words=[]
        text = "M:{:.1f}%, F:{:.1f}%".format(preds[0][0] * 100,preds[0][1] * 100)
        words.append(text)
        img(image)
        keywords({'text':words})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Process', NodeMeta.KEYWORDS: []})
    def image_crop(input=('ImagePin', 0), img=(REF, ('ImagePin', 0)),
                       rect=('GraphElementPin', 0)
                       ):
        """Takes an image and mask and applied logic and operation"""

        if 'rect' in rect.graph.keys():
            only_rect=rect.graph['rect']
            if only_rect:
                x, y, w, h=only_rect[0]
                (ih, iw) = input.image.shape[:2]
                if (x+w)<=iw and (y+h)<ih:
                    img(input.image[y:(y+h),x:(x+w),:])

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Detection', NodeMeta.KEYWORDS: []})
    def blob_detector(input=('ImagePin', 0),
                      thresholdStep=('IntPin',10),
                      minThreshold=('IntPin', 50, {PinSpecifires.VALUE_RANGE: (0, 255)}),
                      maxThreshold=('IntPin', 220 ,{PinSpecifires.VALUE_RANGE: (0, 255)}),
                      minRepeatability=('IntPin', 2),
                      minDistBetweenBlobs=('FloatPin', 10.0),
                      filterByColor=('BoolPin', True),
                      blobColor=('IntPin', 0),
                      filterByArea=('BoolPin', True),
                      minArea=('FloatPin', 25.0),
                      maxArea=('FloatPin', 5000.0),
                      filterByCircularity=('BoolPin', False),
                      minCircularity=('FloatPin', 0.8, {PinSpecifires.VALUE_RANGE: (0, 1.0)}),
                      maxCircularity=('FloatPin', 1, {PinSpecifires.VALUE_RANGE: (0, 1.0)}),
                      filterByInertia=('BoolPin', True),
                      minInertiaRatio=('FloatPin', 0.1,{PinSpecifires.VALUE_RANGE: (0, 1.0)}),
                      maxInertiaRatio=('FloatPin', 1,{PinSpecifires.VALUE_RANGE: (0, 1.0)}),
                      filterByConvexity=('BoolPin', False),
                      minConvexity=('FloatPin', 0.95,{PinSpecifires.VALUE_RANGE: (0, 1.0)}),
                      maxConvexity=('FloatPin', 1,{PinSpecifires.VALUE_RANGE: (0, 1.0)}),
                      maxTotalKeypoints=('IntPin', 1000),
                      gridRows=('IntPin', 4),
                      gridCols=('IntPin', 4),
                      maxLevel=('IntPin', 2),
                      draw_key_points=(REF,('GraphElementPin', 0))
                      ):

        params = cv2.SimpleBlobDetector_Params()
        params.thresholdStep=thresholdStep
        params.minThreshold = minThreshold
        params.maxThreshold = maxThreshold
        params.minDistBetweenBlobs=minDistBetweenBlobs
        params.minRepeatability=minRepeatability

        params.filterByColor=filterByColor
        params.blobColor=blobColor

        params.filterByArea = filterByArea
        params.minArea = minArea
        params.maxArea=maxArea

        params.filterByCircularity = filterByCircularity
        params.minCircularity = minCircularity
        params.maxCircularity=maxCircularity

        params.filterByConvexity = filterByConvexity
        params.minConvexity = minConvexity
        params.maxConvexity=maxConvexity

        params.filterByInertia = filterByInertia
        params.minInertiaRatio = minInertiaRatio
        params.maxInertiaRatio=maxInertiaRatio

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        ret= detector.detect(input.image)
        draw_key_points({'key_point': ret})

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'Process', NodeMeta.KEYWORDS: []})
    def ocr(input=('ImagePin', 0),
            img=(REF, ('ImagePin', 0)),
            engine= ('StringPin', 'LSTM',
                     {PinSpecifires.VALUE_LIST: ["Legacy","LSTM",'Legacy+LSTM','Default']}),
            boxes=('GraphElementPin', 0),
            texts=(REF,('GraphElementPin', 0))
                   ):
        # loop over the bounding boxes to find the coordinate of bounding boxes
        oem_dict={'Legacy':0,
                  'LSTM':1,
                  'Legacy+LSTM': 2,
                  'Default': 3,
                  }
        results=[]
        if 'rect' in boxes.graph:
            for (startX, startY, w, h) in boxes.graph['rect']:
                startX = int(startX)
                startY = int(startY)
                endX = int(startX+w)
                endY = int(startY+h)
                r = input.image[startY:endY, startX:endX]
                configuration = (f"-l eng --oem {oem_dict[engine]} --psm 6")
                text = pytesseract.image_to_string(r, config=configuration)
                if text:
                    d={'box':(startX,startY,w,h),'class':text}
                    results.append(d)
        texts({'detection':results})
        img(input.image)

