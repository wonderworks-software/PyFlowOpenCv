from PyFlow.Core import PinBase
from PyFlow.Core.Common import *
import numpy as np
import json
import cv2

class MyImage():
    def __init__(self, image=None):
        if isinstance(image, MyImage):
            if image.image.__class__.__name__ == "UMat":
                self.image = cv2.UMat(image.image)
            else:
                self.image = image.image.copy()
        elif isinstance(image, np.ndarray) or image.__class__.__name__ == "UMat":
            self.image = image
        else:
            self.image = np.zeros((2, 2, 3), np.uint8)


class NoneEncoder(json.JSONEncoder):
    def default(self, vec3):
        return None


class NoneDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(NoneDecoder, self).__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, vec3Dict):
        return MyImage()

class VideoInput():
    def __init__(self, video_capture=None):
        if isinstance(video_capture, VideoInput):
            self.video_capture=video_capture.video_capture
        elif isinstance(video_capture, cv2.VideoCapture):
            self.video_capture = video_capture
        else:
            self.video_capture=None
    def read(self):
        if isinstance(self.video_capture, cv2.VideoCapture):
            return self.video_capture.read()
        else:
            return None,None


class VideoPin(PinBase):
    """doc string for ImagePin"""

    def __init__(self, name, parent, direction, **kwargs):
        super(VideoPin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue(VideoInput())
        self.disableOptions(PinOptions.Storable)

    @staticmethod
    def jsonEncoderClass():
        return NoneEncoder

    @staticmethod
    def jsonDecoderClass():
        return NoneDecoder

    @staticmethod
    def IsValuePin():
        return True

    @staticmethod
    def supportedDataTypes():
        return ('VideoPin',)

    @staticmethod
    def pinDataTypeHint():
        return 'VideoPin', VideoInput()

    @staticmethod
    def color():
        return (220, 50, 50, 255)

    @staticmethod
    def internalDataStructure():
        return VideoInput

    @staticmethod
    def processData(data):
        # if data.__class__.__name__== "VideoCapture":
        #     return data
        # else:
        #     raise Exception("non Valid VideoCapture")
        return VideoPin.internalDataStructure()(data)

class ImagePin(PinBase):
    """doc string for ImagePin"""

    def __init__(self, name, parent, direction, **kwargs):
        super(ImagePin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue(MyImage())
        self.disableOptions(PinOptions.Storable)

    @staticmethod
    def jsonEncoderClass():
        return NoneEncoder

    @staticmethod
    def jsonDecoderClass():
        return NoneDecoder

    @staticmethod
    def IsValuePin():
        return True

    @staticmethod
    def supportedDataTypes():
        return ('ImagePin',)

    @staticmethod
    def pinDataTypeHint():
        return 'ImagePin', MyImage()

    @staticmethod
    def color():
        return (200, 200, 50, 255)

    @staticmethod
    def internalDataStructure():
        return MyImage

    @staticmethod
    def processData(data):
        return ImagePin.internalDataStructure()(data)

class GraphElement():
    def __init__(self, graph=None):
        if isinstance(graph, GraphElement):
            self.graph=graph.graph
        elif isinstance(graph, dict):
            self.graph = graph
        else:
            self.graph={}

    def draw(self, image):
        if self.graph:
            for draw_type,draw_list in self.graph.items():
                if draw_type=='rect':
                    for (x, y, w, h) in draw_list :
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if draw_type=='point':
                    for (x, y) in draw_list :
                        cv2.circle(image, (int(x), int(y)),5 , (0, 255, 0), -1)
                if draw_type=='key_point':
                    image=cv2.drawKeypoints(image, draw_list, image, (255, 255, 0), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        return image


class GraphElementPin(PinBase):
    """doc string for GraphElement"""

    def __init__(self, name, parent, direction, **kwargs):
        super(GraphElementPin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue({})
        self.disableOptions(PinOptions.Storable)

    @staticmethod
    def jsonEncoderClass():
        return NoneEncoder

    @staticmethod
    def jsonDecoderClass():
        return NoneDecoder

    @staticmethod
    def IsValuePin():
        return True

    @staticmethod
    def supportedDataTypes():
        return ('GraphElementPin',)

    @staticmethod
    def pinDataTypeHint():
        return 'GraphElementPin', {}

    @staticmethod
    def color():
        return (200, 200, 50, 255)

    @staticmethod
    def internalDataStructure():
        return GraphElement

    @staticmethod
    def processData(data):
        return GraphElementPin.internalDataStructure()(data)

class KeyPoints():
    def __init__(self, key_points=None):
        if isinstance(key_points, KeyPoints):
            self.points=np.array(key_points.points)
        elif type(key_points)==tuple:
            self.points = key_points[0]
        elif isinstance(key_points, np.ndarray):
            self.points = np.array(key_points)
        else:
            self.points= None

class KeyPointsPin(PinBase):
    """doc string for KeyPointsPin"""

    def __init__(self, name, parent, direction, **kwargs):
        super(KeyPointsPin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue(KeyPoints())
        self.disableOptions(PinOptions.Storable)

    @staticmethod
    def jsonEncoderClass():
        return NoneEncoder

    @staticmethod
    def jsonDecoderClass():
        return NoneDecoder

    @staticmethod
    def IsValuePin():
        return True

    @staticmethod
    def supportedDataTypes():
        return ('KeyPointsPin',)

    @staticmethod
    def pinDataTypeHint():
        return 'KeyPointsPin', KeyPoints()

    @staticmethod
    def color():
        return (200, 200, 50, 255)

    @staticmethod
    def internalDataStructure():
        return KeyPoints

    @staticmethod
    def processData(data):
        return KeyPointsPin.internalDataStructure()(data)

class BackgroundSubtractor():
    def __init__(self, bgs=None):
        if isinstance(bgs, BackgroundSubtractor):
            self.bgs=bgs.bgs
        elif bgs:
            self.bgs = bgs
        else:
            self.bgs= None

class BackgroundSubtractorPin(PinBase):
    """doc string for KeyPointsPin"""

    def __init__(self, name, parent, direction, **kwargs):
        super(BackgroundSubtractorPin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue(BackgroundSubtractor())
        self.disableOptions(PinOptions.Storable)

    @staticmethod
    def jsonEncoderClass():
        return NoneEncoder

    @staticmethod
    def jsonDecoderClass():
        return NoneDecoder

    @staticmethod
    def IsValuePin():
        return True

    @staticmethod
    def supportedDataTypes():
        return ('BackgroundSubtractorPin',)

    @staticmethod
    def pinDataTypeHint():
        return 'BackgroundSubtractorPin', BackgroundSubtractor()

    @staticmethod
    def color():
        return (200, 200, 50, 255)

    @staticmethod
    def internalDataStructure():
        return BackgroundSubtractor

    @staticmethod
    def processData(data):
        return BackgroundSubtractorPin.internalDataStructure()(data)

class Descriptor():
    def __init__(self, desc=None):
        if isinstance(desc, Descriptor):
            self.desc=desc.desc
        elif isinstance(desc, np.ndarray):
            self.desc = desc
        elif desc is not None:
            self.desc = desc
        else:
            self.desc= None

class DescriptorPin(PinBase):
    """doc string for KeyPointsPin"""

    def __init__(self, name, parent, direction, **kwargs):
        super(DescriptorPin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue(Descriptor())
        self.disableOptions(PinOptions.Storable)

    @staticmethod
    def jsonEncoderClass():
        return NoneEncoder

    @staticmethod
    def jsonDecoderClass():
        return NoneDecoder

    @staticmethod
    def IsValuePin():
        return True

    @staticmethod
    def supportedDataTypes():
        return ('DescriptorPin',)

    @staticmethod
    def pinDataTypeHint():
        return 'DescriptorPin', Descriptor()

    @staticmethod
    def color():
        return (200, 200, 50, 255)

    @staticmethod
    def internalDataStructure():
        return Descriptor

    @staticmethod
    def processData(data):
        return DescriptorPin.internalDataStructure()(data)

class FeatureMatch():
    def __init__(self, match=None):
        if isinstance(match, FeatureMatch):
            self.match=match.match
        elif isinstance(match, np.ndarray):
            self.match = match
        elif match:
            self.match = match
        else:
            self.match= None

class FeatureMatchPin(PinBase):
    """doc string for KeyPointsPin"""

    def __init__(self, name, parent, direction, **kwargs):
        super(FeatureMatchPin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue(FeatureMatch())
        self.disableOptions(PinOptions.Storable)

    @staticmethod
    def jsonEncoderClass():
        return NoneEncoder

    @staticmethod
    def jsonDecoderClass():
        return NoneDecoder

    @staticmethod
    def IsValuePin():
        return True

    @staticmethod
    def supportedDataTypes():
        return ('FeatureMatchPin',)

    @staticmethod
    def pinDataTypeHint():
        return 'FeatureMatchPin', FeatureMatch()

    @staticmethod
    def color():
        return (200, 200, 50, 255)

    @staticmethod
    def internalDataStructure():
        return FeatureMatch

    @staticmethod
    def processData(data):
        return FeatureMatchPin.internalDataStructure()(data)

