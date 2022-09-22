from Qt import QtCore , QtWidgets , QtGui
from PyFlow.UI.Canvas.UICommon import *
from PyFlow.Core.Common import push,getConnectedPins
from PyFlow.Packages.PyFlowOpenCv.UI.UIOpenCvBaseNode import UIOpenCvBaseNode
from PyFlow.Packages.PyFlowOpenCv.CV_classes.imageUtils import *
from PyFlow.Packages.PyFlowOpenCv.UI.pc_ImageCanvasWidget import toQImage

import numpy as np

class SignalEmiter(QtCore.QObject):
    """docstring for SignalEmiter"""
    imagePainted = QtCore.Signal(QtGui.QPixmap)
    def __init__(self):
        super(SignalEmiter, self).__init__()

class PainterWidget(QtWidgets.QGraphicsPixmapItem):
    """A widget where user can draw with their mouse
    """
    _MANIP_MODE_NONE = 0
    _MANIP_MODE_PAINT = 1
    _MANIP_MODE_ERASE = 2

    imagePainted = QtCore.Signal(str)
    def __init__(self, parent=None,Node=None,useInitImage=False):
        super(PainterWidget, self).__init__(parent)
        self.SignalEmiter = SignalEmiter()

        self.Node = Node
        self.previous_pos = None
        self.painter = QtGui.QPainter()
        self.DrawingPen = QtGui.QPen(QtCore.Qt.white)
        self.DrawingPen.setWidth(10)
        self.DrawingPen.setCapStyle(QtCore.Qt.RoundCap)
        self.DrawingPen.setJoinStyle(QtCore.Qt.RoundJoin)
        self._manipulationMode = self._MANIP_MODE_NONE
        
        #self.setFlag(QtWidgets.QGraphicsWidget.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsWidget.ItemIsFocusable)
        self.setFlag(QtWidgets.QGraphicsWidget.ItemIsSelectable)
        self.setFlag(QtWidgets.QGraphicsWidget.ItemSendsGeometryChanges)
        self.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)        
        self.setAcceptHoverEvents(True)
        self.mask_image = None
        self.useInitImage = useInitImage

    def setInitImage(self,init_image):
        self.useInitImage = True
        self.setPixmap(init_image)
        self.mask_image = QtGui.QPixmap(init_image)
        self.mask_image.fill(QtCore.Qt.transparent)

    def mousePressEvent(self, event):
        modifiers = event.modifiers()
        if event.button() == QtCore.Qt.LeftButton and modifiers in [QtCore.Qt.NoModifier,QtCore.Qt.ShiftModifier]:
            self._manipulationMode = self._MANIP_MODE_PAINT
            self.previous_pos = self.mapToScene(event.pos())
            #self.DrawingPen.setColor(QtCore.Qt.white)
        elif event.button() == QtCore.Qt.LeftButton and modifiers in [QtCore.Qt.ControlModifier]:
            self._manipulationMode = self._MANIP_MODE_ERASE
            #self.DrawingPen.setColor(QtGui.QColor(0,0,0,255))
            self.previous_pos = self.mapToScene(event.pos())
        super(PainterWidget, self).mousePressEvent( event)

    def mouseMoveEvent(self, event):
        pix = self.pixmap()
        self.painter.begin(pix)        
        #TODO:: ADD BRUSHES
        if self._manipulationMode == self._MANIP_MODE_PAINT:
            current_pos = self.mapToScene(event.pos())
            self.painter.setRenderHints(QtGui.QPainter.Antialiasing, True)
            self.painter.setPen(self.DrawingPen)
            self.painter.drawLine(self.previous_pos, current_pos)

        if self._manipulationMode == self._MANIP_MODE_ERASE:
            r = QtCore.QRect(QtCore.QPoint(), self.DrawingPen.width()*QtCore.QSize())
            r.moveCenter(self.mapToScene(event.pos()).toPoint())
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            self.painter.eraseRect(r)

        self.painter.end()
        self.setPixmap(pix)

        self.previous_pos = current_pos

        super(PainterWidget, self).mouseMoveEvent( event)

    def mouseReleaseEvent(self, event):
        self._manipulationMode = self._MANIP_MODE_NONE
        self.previous_pos = None
        if self.useInitImage:
            self.SignalEmiter.imagePainted.emit(self.mask_image)
        else:
            self.SignalEmiter.imagePainted.emit(self.pixmap())
        super(PainterWidget, self).mouseReleaseEvent( event)

    def clear(self):
        """ Clear the pixmap """
        self.pixmap.fill(QtCore.Qt.transparent)
        self.update()

def QPixmapToArray(pixmap):
    ## Get the size of the current pixmap
    size = pixmap.size()
    h = size.width()
    w = size.height()

    ## Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits().tobytes()

    ## Using the np.frombuffer function to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w,h,4))

    return img

def join_pixmap(p1, p2, mode=QtGui.QPainter.CompositionMode_SourceOver):
    s = p1.size().expandedTo(p2.size())
    result =  QtGui.QPixmap(s)
    result.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(result)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.drawPixmap(QtCore.QPoint(), p1)
    painter.setCompositionMode(mode)
    painter.drawPixmap(result.rect(), p2, p2.rect())
    painter.end()
    return result

class UICv_PaintMask(UIOpenCvBaseNode):
    def __init__(self, raw_node):
        super(UICv_PaintMask, self).__init__(raw_node)
        self.sizeXPin = self._rawNode.getPinByName("sizeX")
        self.sizeYPin = self._rawNode.getPinByName("sizeY")
        self.imageRefPin = self._rawNode.getPinByName("imageRef")
        self.imgPin = self._rawNode.getPinByName("img")
        self.rgbMaskPin = self._rawNode.getPinByName("rgbMask")

        self.burshShizePin = self._rawNode.getPinByName("BrushShize")

        self.sizeXPin.dataBeenSet.connect(self.updateCanvasSize)
        self.sizeYPin.dataBeenSet.connect(self.updateCanvasSize)
        self.imageRefPin.dataBeenSet.connect(self.updateCanvasSize)
        self.burshShizePin.dataBeenSet.connect(self.updateburshShize)

        self.Painter = PainterWidget(Node=self)
        self.Painter.setShapeMode(QtWidgets.QGraphicsPixmapItem.BoundingRectShape)
        self.Painter.SignalEmiter.imagePainted.connect(self.updateimg)

        pix = QtGui.QPixmap(self.sizeXPin.getData(),self.sizeYPin.getData())
        pix.fill(QtCore.Qt.transparent)
        self.Painter.setPixmap(pix)

        self.openProperties = []

    def updateburshShize(self,pin):
        self.Painter.DrawingPen.setWidth(self.burshShizePin.getData())

    def updateCanvasSize(self, pin) :
        if len(self.imageRefPin.affected_by) == 0:
            n_w, n_h = (self.sizeXPin.getData(),self.sizeYPin.getData())
        else:
            n_h, n_w, _ = get_h_w_c(self.imageRefPin.getData())
        o_h, o_w, _ = get_h_w_c(self._rawNode.IMAGE)
        h = min(o_h-1,max(n_h,1))
        w = min(o_w-1,max(n_w,1))
        self._rawNode.IMAGE = self._rawNode.IMAGE[0:(h),0:(w),:]     
        self._rawNode.IMAGE = expand_image_to_fit_rect(self._rawNode.IMAGE,(max(n_w,1), max(n_h,1)))
        pix = QtGui.QPixmap.fromImage(toQImage(self._rawNode.IMAGE))
        self.Painter.setPixmap(pix)

    def updateimg(self,pixmap):
        self._rawNode.IMAGE = QPixmapToArray(pixmap)
        push(self.imgPin)
        push(self.rgbMaskPin)

    def createInputWidgets(self, inputsCategory, inGroup=None, pins=True):
        
        inputsCategory.destroyed.connect(lambda x=None: self.removeItemFromViewer(inputsCategory))
        self.openProperties.append(inputsCategory)
        preIndex = inputsCategory.Layout.count()
        if pins:
            super(UICv_PaintMask, self).createInputWidgets(inputsCategory, inGroup)        

        instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
        if self.Painter.scene() == None:
            instance.viewer._scene.addItem(self.Painter)

        sizeXwidg = inputsCategory.getWidgetByName("sizeX")
        sizeYwidg = inputsCategory.getWidgetByName("sizeY")

    def removeItemFromViewer(self,obj):
        if obj in self.openProperties:
            self.openProperties.remove(obj)
        #if len(self.openProperties)==0 and self.item.scene():
        instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
        instance.viewer._scene.removeItem(self.Painter)