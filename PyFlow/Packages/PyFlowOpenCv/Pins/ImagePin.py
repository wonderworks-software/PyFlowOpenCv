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
                    image=cv2.drawKeypoints(image, draw_list, image, (255, 0, 0), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
                if draw_type == 'text':
                    for text in draw_list:
                        image= cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if draw_type == 'detection':
                    for text in draw_list:
                        image_height, image_width, _ = image.shape
                        (x,y,w,h)=text['box']
                        class_name=text['class']
                        confidence=str(round(text['confidence']*100))+'%' if 'confidence' in text else ''
                        fontSize = image_height / 1024.0
                        label = (str(class_name) + " " + confidence).upper()
                        color = (255,0,0)
                        x1 = max(int(x), 0)
                        y1 = max(int(y), 0)
                        x2=int(x+w)
                        y2=int(y+h)
                        thickness=max(round(4*fontSize),1)
                        image=cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                        image=cv2.rectangle(image, (x1-thickness, y1-thickness), (x2-thickness, y2-thickness), (255,255,255), thickness)
                        (textWidth, textHeight), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontSize,
                                                                            thickness)
                        if y1 - textHeight <= 0:
                            y1 = y1 + textHeight
                        image=cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), thickness)
                        image=cv2.putText(image, label, (x1-thickness, y1-thickness), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255),thickness)
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


class NumpyData():
    def __init__(self, data=None):
        if isinstance(data, NumpyData):
            self.data=data.data
        elif isinstance(data, np.ndarray):
            self.data= data
        elif data:
            self.data= data
        else:
            self.data= None


class NumpyDataPin(PinBase):
    """doc string for NumpyDataPin"""

    def __init__(self, name, parent, direction, **kwargs):
        super(NumpyDataPin, self).__init__(name, parent, direction, **kwargs)
        self.setDefaultValue(NumpyData())
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
        return ('NumpyDataPin',)

    @staticmethod
    def pinDataTypeHint():
        return 'NumpyDataPin', NumpyData()

    @staticmethod
    def color():
        return (200, 200, 50, 255)

    @staticmethod
    def internalDataStructure():
        return NumpyData

    @staticmethod
    def processData(data):
        return NumpyDataPin.internalDataStructure()(data)

class KeyPoints(NumpyData):
    pass

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

class BackgroundSubtractor(NumpyData):
    pass

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

class Descriptor(NumpyData):
    pass

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

class FeatureMatch(NumpyData):
    pass

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

