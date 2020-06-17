from PyFlow.Core import NodeBase
from PyFlow.Core.NodeBase import NodePinsSuggestionsHelper
from PyFlow.Core.Common import *
from Qt import QtWidgets

class ViewerNode(NodeBase):
    def __init__(self, name):
        super(ViewerNode, self).__init__(name)
        self.inExec = self.createInputPin(DEFAULT_IN_EXEC_NAME, 'ExecPin', None, self.compute)
        self.inp = self.createInputPin('img', 'ImagePin')
        self.arrayData = self.createInputPin('data', 'GraphElementPin', structure=StructureType.Array)
        self.arrayData.enableOptions(PinOptions.AllowMultipleConnections)
        self.outExec = self.createOutputPin(DEFAULT_OUT_EXEC_NAME, 'ExecPin')

    @staticmethod
    def pinTypeHints():
        helper = NodePinsSuggestionsHelper()
        helper.addInputDataType('ExecPin')
        helper.addInputDataType('ImagePin')
        helper.addInputDataType('GraphElementPin')
        helper.addOutputDataType('ExecPin')
        helper.addInputStruct(StructureType.Multi)
        helper.addInputStruct(StructureType.Single)
        helper.addInputStruct(StructureType.Array)
        helper.addOutputStruct(StructureType.Single)
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
        if self.inp.dirty or self.arrayData.dirty:
            inputData = self.inp.getData()
            instance = self._wrapper.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
            yInputPins= sorted(self.arrayData.affected_by, key=lambda pin: pin.owningNode().y)
            draw_image=inputData.image
            for i in yInputPins:
                draw_image=i.getData().draw(draw_image)
            instance.viewer.setNumpyArray(draw_image)
            self.inp.setClean()
            QtWidgets.QApplication.processEvents()
        self.outExec.call()
