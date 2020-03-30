import cv2

from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Core.Common import *

class ViewerNode(NodeBase):
    def __init__(self, name):
        super(ViewerNode, self).__init__(name)
        self.inExec = self.createInputPin(DEFAULT_IN_EXEC_NAME, 'ExecPin', None, self.compute)
        self.inp = self.createInputPin('img', 'ImagePin')

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('ImagePin')
        helper.addInputStruct(StructureType.Single)
        return helper

    @staticmethod
    def category():
        return 'cv'

    @staticmethod
    def keywords():
        return []

    @staticmethod
    def description():
        return "Description in rst format."

    def compute(self, *args, **kwargs):
        inputData = self.inp.getData()
        instance = self._wrapper.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
        if len(inputData.image.shape)==2:
            instance.viewer.setNumpyArray(cv2.cvtColor(inputData.image, cv2.COLOR_GRAY2BGR))
        else:
            instance.viewer.setNumpyArray(inputData.image)
