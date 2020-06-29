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
    xCenterChanged = QtCore.Signal(int)
    yCenterChanged = QtCore.Signal(int)
    rotationChanged = QtCore.Signal(float)
    rotationUpdated = QtCore.Signal(float)
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
            self.rotationChanged.emit(-(self.angle + angl))
            #print angl
        self.updateCursor(event.pos().x())

    def setX(self,Pos):
        self.setPos(Pos-50,self.pos().y())

    def setY(self,Pos):
        self.setPos(self.pos().x(),Pos-50)

    def rotate(self,angle,override=False):
        if not self._manipulationMode == self._MANIP_MODE_ROT or override:
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

        self.anglePin.dataBeenSet.connect(self.updateWidgetRotation)
        self.xCenterPin.dataBeenSet.connect(self.updateWidgetXposition)
        self.yCenterPin.dataBeenSet.connect(self.updateWidgetYposition)

        self.openProperties = []

    def updateWidgetRotation(self,pin):
        self.item.rotate(pin.getData())
    def updateWidgetXposition(self,pin):
        self.item.setX(pin.getData())
    def updateWidgetYposition(self,pin):
        self.item.setY(pin.getData())

    def setAngle(self,angle):
        if len(self.anglePin.affected_by) == 0:
            self.anglePin.setData(angle)
            self.item.rotationUpdated.emit(angle)
        else:
            self.item.rotate(self.anglePin.getData(),True)
            self.item.rotationUpdated.emit(self.anglePin.getData())
        self._refresh_tool_image()

    def setCenter(self,center):
        if len(self.xCenterPin.affected_by) == 0:
            self.xCenterPin.setData(int(center.x()))
            self.item.xCenterChanged.emit(int(center.x()))         
        else:
            self.item.setX(self.xCenterPin.getData())
            self.item.xCenterChanged.emit(self.xCenterPin.getData())
        if len(self.yCenterPin.affected_by) == 0:
            self.yCenterPin.setData(int(center.y()))
            self.item.yCenterChanged.emit(int(center.y()))   
        else:
            self.item.setY(self.yCenterPin.getData())
            self.item.yCenterChanged.emit(self.yCenterPin.getData())
        self._refresh_tool_image()

    def _refresh_tool_image(self):
        self.refreshImage()
        instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv", "ImageViewerTool")
        if instance:
            img = self.imagePin.getData()
            instance.viewer.setNumpyArray(img.image)

    def createInputWidgets(self, inputsCategory, inGroup=None, pins=True):
        
        inputsCategory.destroyed.connect(lambda x=None: self.removeItemFromViewer(inputsCategory))
        self.openProperties.append(inputsCategory)
        preIndex = inputsCategory.Layout.count()
        if pins:
            super(UICv_TransformNode, self).createInputWidgets(inputsCategory, inGroup)        

        instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
        if self.item.scene() == None:
            instance.viewer._scene.addItem(self.item)

        angleWidg = inputsCategory.getWidgetByName("angle")
        xCenterWidg = inputsCategory.getWidgetByName("xCenter")
        yCenterWidg = inputsCategory.getWidgetByName("yCenter")

        self.item.rotationUpdated.connect(angleWidg.setWidgetValue)
        self.item.xCenterChanged.connect(xCenterWidg.setWidgetValue)
        self.item.yCenterChanged.connect(yCenterWidg.setWidgetValue)
        #self.angleWidg.sb.valueChanged.connect(self.item.rotate)
        #self.xCenterWidg.sb.valueChanged.connect(self.item.setX)
        #self.yCenterWidg.sb.valueChanged.connect(self.item.setY)

        self.item.rotate(self.anglePin.getData())
        self.item.setX(self.xCenterPin.getData())
        self.item.setY(self.yCenterPin.getData())

    def removeItemFromViewer(self,obj):
        if obj in self.openProperties:
            self.openProperties.remove(obj)
        if len(self.openProperties)==0 and self.item.scene():
            instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
            instance.viewer._scene.removeItem(self.item)