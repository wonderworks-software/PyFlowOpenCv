from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Core.Common import *
from Qt import QtWidgets
import numpy as np

class PaintMask(NodeBase):
    def __init__(self, name):
        super(PaintMask, self).__init__(name)
        #self.inExec = self.createInputPin(DEFAULT_IN_EXEC_NAME, 'ExecPin', None, self.compute)
        #self.clearimg = self.createInputPin("clearImage", 'ExecPin', None, self.clearImage)

        self.imageRef = self.createInputPin('imageRef', 'ImagePin')
        self.sizeX = self.createInputPin('sizeX', 'IntPin',512)
        self.sizeY = self.createInputPin('sizeY', 'IntPin',512)

        self.penSize = self.createInputPin('BrushShize', 'FloatPin',10)

        self.img = self.createOutputPin('img', 'ImagePin')
        self.rgbMask = self.createOutputPin('rgbMask', 'ImagePin')
        #self.outExec = self.createOutputPin(DEFAULT_OUT_EXEC_NAME, 'ExecPin')

        self.IMAGE = np.zeros((512, 512, 4), np.uint8)

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('ExecPin')
        helper.addInputDataType('ImagePin')
        helper.addInputDataType('GraphElementPin')
        helper.addOutputDataType('ExecPin')
        helper.addInputStruct(StructureType.Single)
        return helper

    @staticmethod
    def category():
        return 'Viewers'

    @staticmethod
    def keywords():
        return []

    @staticmethod
    def description():
        return "Description in rst format."

    def compute(self, *args, **kwargs):
        self.img.setData(self.IMAGE)
        self.rgbMask.setData(self.IMAGE[:,:,:3])
