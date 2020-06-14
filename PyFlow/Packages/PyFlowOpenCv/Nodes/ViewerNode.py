import cv2

from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Core.Common import *

class ViewerNode(NodeBase):
    def __init__(self, name):
        super(ViewerNode, self).__init__(name)
        self.inExec = self.createInputPin(DEFAULT_IN_EXEC_NAME, 'ExecPin', None, self.compute)
        self.inp = self.createInputPin('img', 'ImagePin')
        self.inRect = self.createInputPin('rects', 'AnyPin')

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('ImagePin')
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
        if self.inp.dirty or self.inRect.dirty:
            inputData = self.inp.getData()
            inputRect= self.inRect.getData()
            instance = self._wrapper.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
            if inputRect:
                for (x, y, w, h) in inputRect:
                    cv2.rectangle(inputData.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            instance.viewer.setNumpyArray(inputData.image)
