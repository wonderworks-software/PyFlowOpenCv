import cv2
from Qt import QtGui,QtCore
from PyFlow.UI import RESOURCES_DIR
from PyFlow.UI.Canvas.UINodeBase import UINodeBase
from PyFlow.UI.Canvas.NodeActionButton import NodeActionButtonBase
from PyFlow.UI.Canvas.UICommon import *
from Qt.QtWidgets import QLabel
import os

class ViewImageNodeActionButton(NodeActionButtonBase):
    """docstring for ViewImageNodeActionButton."""
    def __init__(self, svgFilePath, action, uiNode):
        super(ViewImageNodeActionButton, self).__init__(svgFilePath, action, uiNode)
        self.svgIcon.setElementId("Expand")

    def mousePressEvent(self, event):
        super(ViewImageNodeActionButton, self).mousePressEvent(event)
        if not self.parentItem().displayImage:
            self.svgIcon.setElementId("Expand")
        else:
            self.svgIcon.setElementId("Collapse")

class UIOpenCvBaseNode(UINodeBase):
    def __init__(self, raw_node):
        super(UIOpenCvBaseNode, self).__init__(raw_node)
        self.actionViewImage = self._menu.addAction("ViewImage")
        self.actionViewImage.triggered.connect(self.viewImage)
        self.actionViewImage.setData(NodeActionButtonInfo(os.path.dirname(__file__)+"/resources/ojo.svg", ViewImageNodeActionButton))
        self.actionRefreshImage = self._menu.addAction("RefreshImage")
        self.actionRefreshImage.triggered.connect(self.refreshImage)
        self.actionRefreshImage.setData(NodeActionButtonInfo(os.path.dirname(__file__)+"/resources/reload.svg", NodeActionButtonBase))        
        self.displayImage = False
        self.resizable = True
        self.Imagelabel = QLabel("noImage")
        self.pixmap = QtGui.QPixmap()
        self.addWidget(self.Imagelabel)
        self.Imagelabel.setVisible(False)
        self.updateSize()
        self._rawNode.computed.connect(self.updateImage)

    @property
    def collapsed(self):
        return self._collapsed

    @collapsed.setter
    def collapsed(self, bCollapsed):
        if bCollapsed != self._collapsed:
            self._collapsed = bCollapsed
            self.aboutToCollapse(self._collapsed)
            for i in range(0, self.inputsLayout.count()):
                inp = self.inputsLayout.itemAt(i)
                inp.setVisible(not bCollapsed)
            for o in range(0, self.outputsLayout.count()):
                out = self.outputsLayout.itemAt(o)
                out.setVisible(not bCollapsed)
            for cust in range(0, self.customLayout.count()):
                out = self.customLayout.itemAt(cust)
                out.setVisible(not bCollapsed)
            if not self.displayImage:
                self.Imagelabel.setVisible(False)
            self.updateNodeShape()

    def updateImage(self,*args, **kwargs):
        if self.displayImage and not self.collapsed :
            pin = self._rawNode.getPinByName("img")
            if pin:
                img = pin.getData()
                if len(img.image.shape) == 2:
                    self.setNumpyArray(cv2.cvtColor(img.image, cv2.COLOR_GRAY2BGR))
                else:
                    self.setNumpyArray(img.image)

    def refreshImage(self):
        if self.displayImage and not self.collapsed :
            pin = self._rawNode.getPinByName("img")
            if pin:
                self._rawNode.processNode()
                img = pin.getData()
                if len(img.image.shape) == 2:
                    self.setNumpyArray(cv2.cvtColor(img.image, cv2.COLOR_GRAY2BGR))
                else:
                    self.setNumpyArray(img.image)
            self.Imagelabel.setVisible(True)
        else:
            self.Imagelabel.setVisible(False)

    def viewImage(self):
        self.displayImage = not self.displayImage
        self.refreshImage()
        self.updateSize()

    def createQimagefromNumpyArray(self,img):
        i = QtGui.QImage( img, img.shape[ 1 ], img.shape[ 0 ],img.shape[ 1 ] * img.shape[ 2 ],QtGui.QImage.Format_RGB888 )
        return  i

    def setNumpyArray(self,image):
        image = self.createQimagefromNumpyArray(image)
        self.pixmap = QtGui.QPixmap.fromImage(image,QtCore.Qt.ThresholdAlphaDither)
        self.updateSize()

    def paint(self, painter, option, widget):
        self.updateSize()
        super(UIOpenCvBaseNode, self).paint(painter, option, widget)

    def updateSize(self):
        if not self.pixmap.isNull():
            scaledPixmap = self.pixmap.scaledToWidth(self.customLayout.geometry().width())
            self.Imagelabel.setPixmap(scaledPixmap)
