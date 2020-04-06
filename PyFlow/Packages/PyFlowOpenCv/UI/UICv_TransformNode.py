from Qt import QtGui,QtCore,QtWidgets
from PyFlow.UI.Canvas.UICommon import *
from PyFlow.Packages.PyFlowOpenCv.UI.UIOpenCvBaseNode import UIOpenCvBaseNode
import os
import math
class TransformItem(QtWidgets.QGraphicsWidget):
    """docstring for TransformItem"""
    _MANIP_MODE_NONE = 0
    _MANIP_MODE_MOVE = 1
    _MANIP_MODE_ROT = 2
    centerChanged = QtCore.Signal(QtCore.QPointF)
    rotationChanged = QtCore.Signal(float)

    def __init__(self,parent=None):
        super(TransformItem, self).__init__(parent=parent)
        self.setFlag(QtWidgets.QGraphicsWidget.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsWidget.ItemIsFocusable)
        self.setFlag(QtWidgets.QGraphicsWidget.ItemIsSelectable)
        self.setFlag(QtWidgets.QGraphicsWidget.ItemSendsGeometryChanges)
        self.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)        
        self.setAcceptHoverEvents(True)
        self.rotationCursor = QtGui.QCursor(os.path.dirname(__file__)+"/resources/rotateIcon.png")
        self.pen =  QtGui.QPen(QtCore.Qt.white,2)
        self._manipulationMode = self._MANIP_MODE_NONE

        self.mousePressPose = QtCore.QPointF(0, 0)
        self.mousePos = QtCore.QPointF(0, 0)
        self._lastMousePos = QtCore.QPointF(0, 0)
        self._lastRect = self.boundingRect().center()
        self.angle = 0
        self.setTransformOriginPoint(QtCore.QPoint(50,50))
    def updateCursor(self,pos):
        if pos > 110:
            self.setCursor(self.rotationCursor)
        else:
            self.setCursor(QtCore.Qt.SizeAllCursor)

    def hoverEnterEvent(self, event):
        super(TransformItem, self).hoverEnterEvent(event)
        self.pen =  QtGui.QPen(QtCore.Qt.yellow,4)
        self.updateCursor(event.pos().x())

    def hoverMoveEvent(self, event):
        super(TransformItem, self).hoverMoveEvent(event)
        self.pen =  QtGui.QPen(QtCore.Qt.yellow,4)
        self.updateCursor(event.pos().x())

    def hoverLeaveEvent(self, event):
        super(TransformItem, self).hoverLeaveEvent(event)
        self.pen =  QtGui.QPen(QtCore.Qt.white,2)

    def mousePressEvent(self, event):
        super(TransformItem, self).mousePressEvent(event)
        self.updateCursor(event.pos().x())
        if event.pos().x() <= 110:
            self._manipulationMode = self._MANIP_MODE_MOVE
        else:
            self._manipulationMode = self._MANIP_MODE_ROT
        self._lastMousePos = self.mapToScene(event.pos())
        self._lastRect = self.boundingRect().center()
    
    def mouseReleaseEvent(self,event):
        super(TransformItem, self).mouseReleaseEvent(event)
        self._manipulationMode = self._MANIP_MODE_NONE
        self.angle = self.rotation()

    def mouseMoveEvent(self, event):
        if self._manipulationMode == self._MANIP_MODE_MOVE:
            super(TransformItem, self).mouseMoveEvent(event)
            self.centerChanged.emit(self.mapToScene(QtCore.QPoint(50,50)))
        elif self._manipulationMode == self._MANIP_MODE_ROT:
            prevVector = self._lastMousePos - self.mapToScene(self._lastRect)
            currVector = self.mapToScene(event.pos()) - self.mapToScene(self._lastRect)
            angl = self.getAngle(prevVector,currVector)
            self.setRotation(self.angle + angl)
            self.rotationChanged.emit(self.angle + angl)
            #print angl
        self.updateCursor(event.pos().x())

    def setX(self,Pos):
        self.setPos(Pos-50,self.pos().y())

    def setY(self,Pos):
        self.setPos(self.pos().x(),Pos-50)

    def rotate(self,angle):
        if not self._manipulationMode == self._MANIP_MODE_ROT:
            self.angle = -angle
            self.setRotation(-angle)

    def shape(self):
        rect = self.boundingRect().marginsRemoved(QtCore.QMargins(10,10,60,10))
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.WindingFill)
        path.addEllipse(rect)
        path.addRect(rect.x(),rect.center().y()-5,rect.width()+50,10)
        return path

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super(TransformItem, self).setGeometry(rect)
        self.setPos(rect.topLeft())

    def sizeHint(self, which, constraint):
        return QtCore.QSizeF(150, 100)

    def paint(self, painter, option, widget):
        painter.setPen(self.pen)
        rect = self.boundingRect().marginsRemoved(QtCore.QMargins(10,10,60,10))
        path = QtGui.QPainterPath()
        path.addEllipse(rect)
        path.addRect(rect.center().x(),rect.y(),0,rect.height())
        path.addRect(rect.x(),rect.center().y(),rect.width()+50,0)
        painter.drawPath(path)

    def getAngle(self,A,B):
        ang = math.atan2( A.x()*B.y() - A.y()*B.x(), A.x()*B.x() + A.y()*B.y() )
        return math.degrees(ang)

class UICv_TransformNode(UIOpenCvBaseNode):
    def __init__(self, raw_node):
        super(UICv_TransformNode, self).__init__(raw_node)
        self.xCenterPin = self._rawNode.getPinByName("xCenter")
        self.yCenterPin = self._rawNode.getPinByName("yCenter")
        self.anglePin = self._rawNode.getPinByName("angle")
        self.scalePin = self._rawNode.getPinByName("scale")
        self.item = TransformItem()
        self.item.rotationChanged.connect(self.setAngle)
        self.item.centerChanged.connect(self.setCenter)
        self.angleWidg = None
        self.xCenterWidg = None
        self.yCenterWidg = None

    def setAngle(self,angle):
        self.anglePin.setData(-angle)
        if self.angleWidg:
            self.angleWidg.setWidgetValue(-angle)

    def setCenter(self,center):
        self.xCenterPin.setData(center.x())
        self.yCenterPin.setData(center.y())
        if self.xCenterWidg:
            self.xCenterWidg.setWidgetValue(int(center.x()))
        if self.yCenterWidg:
            self.yCenterWidg.setWidgetValue(int(center.y()))

    def createInputWidgets(self, inputsCategory, inGroup=None, pins=True):
        inputsCategory.destroyed.connect(self.removeItemFromViewer)
        preIndex = inputsCategory.Layout.count()
        if pins:
            super(UICv_TransformNode, self).createInputWidgets(inputsCategory, inGroup)        

        instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
        if self.item not in instance.viewer._scene.items():
            instance.viewer._scene.addItem(self.item)

        self.angleWidg = inputsCategory.getWidgetByName("angle")
        self.xCenterWidg = inputsCategory.getWidgetByName("xCenter")
        self.yCenterWidg = inputsCategory.getWidgetByName("yCenter")
        self.angleWidg.sb.valueChanged.connect(self.item.rotate)
        self.xCenterWidg.sb.valueChanged.connect(self.item.setX)
        self.yCenterWidg.sb.valueChanged.connect(self.item.setY)

        self.item.rotate(self.anglePin.getData())
        self.item.setPos(self.xCenterPin.getData()-50,self.yCenterPin.getData()-50)

    def removeItemFromViewer(self):
        instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
        instance.viewer._scene.removeItem(self.item)