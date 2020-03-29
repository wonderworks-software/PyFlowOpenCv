from nine import str
from Qt import QtGui

from PyFlow.UI.Tool.Tool import DockTool
from PyFlow.Packages.PyFlowOpenCv.UI.pc_ImageCanvasWidget import pc_ImageCanvas

class ImageViewerTool(DockTool):
    """docstring for History tool."""
    def __init__(self):
        super(ImageViewerTool, self).__init__()
        self.viewer = pc_ImageCanvas(self)
        self.setWidget(self.viewer)

    @staticmethod
    def getIcon():
        return QtGui.QIcon(":brick.png")

    @staticmethod
    def toolTip():
        return "My awesome dock tool!"

    @staticmethod
    def isSingleton():
        return True

    @staticmethod
    def name():
        return str("ImageViewerTool")
