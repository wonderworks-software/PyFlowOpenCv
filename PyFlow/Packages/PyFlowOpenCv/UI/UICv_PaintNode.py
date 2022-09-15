from Qt import QtCore , QtWidgets , QtGui
from PyFlow.UI.Canvas.UICommon import *
from PyFlow.Core.Common import push
from PyFlow.Packages.PyFlowOpenCv.UI.UIOpenCvBaseNode import UIOpenCvBaseNode
import numpy as np

class SignalEmiter(QtCore.QObject):
    """docstring for SignalEmiter"""
    imagePainted = QtCore.Signal(QtGui.QPixmap)
    def __init__(self):
        super(SignalEmiter, self).__init__()

class PainterWidget(QtWidgets.QGraphicsPixmapItem):
    """A widget where user can draw with their mouse

    The user draws on a QtGui.QPixmap which is itself paint from paintEvent()

    """
    _MANIP_MODE_NONE = 0
    _MANIP_MODE_PAINT = 1

    imagePainted = QtCore.Signal(str)
    def __init__(self, parent=None,Node=None,useInitImage=False):
        super(PainterWidget, self).__init__(parent)
        self.SignalEmiter = SignalEmiter()

        self.Node = Node
        self.previous_pos = None
        self.painter = QtGui.QPainter()
        self.pen = QtGui.QPen()
        self.pen.setWidth(10)
        self.pen.setCapStyle(QtCore.Qt.RoundCap)
        self.pen.setJoinStyle(QtCore.Qt.RoundJoin)
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
        self.mask_image.fill(QtCore.Qt.white)

    def mousePressEvent(self, event):
        """Override from QtWidgets.QWidget

        Called when user clicks on the mouse

        """
        modifiers = event.modifiers()
        if event.button() == QtCore.Qt.LeftButton:
            self._manipulationMode = self._MANIP_MODE_PAINT
            self.previous_pos = self.mapToScene(event.pos())

        super(PainterWidget, self).mousePressEvent( event)

    def mouseMoveEvent(self, event):
        """Override method from QtWidgets.QWidget

        Called when user moves and clicks on the mouse

        """
        if self._manipulationMode == self._MANIP_MODE_PAINT:
            #print(self.Node.item.scene().parent()._manipulationMode)
            current_pos = self.mapToScene(event.pos())
            pix = self.pixmap()
            self.painter.begin(pix)
            self.painter.setRenderHints(QtGui.QPainter.Antialiasing, True)
            self.painter.setPen(self.pen)
            self.painter.drawLine(self.previous_pos, current_pos)
            self.painter.end()
            self.setPixmap(pix)
            if self.useInitImage:
                self.painter.begin(self.mask_image)
                self.painter.setRenderHints(QtGui.QPainter.Antialiasing, True)
                self.painter.setPen(self.pen)
                self.painter.drawLine(self.previous_pos, current_pos)
                self.painter.end()              

            self.previous_pos = current_pos
            #self.update()

        super(PainterWidget, self).mouseMoveEvent( event)

    def mouseReleaseEvent(self, event):
        """Override method from QtWidgets.QWidget

        Called when user releases the mouse

        """
        self._manipulationMode = self._MANIP_MODE_NONE
        self.previous_pos = None
        if self.useInitImage:
            self.SignalEmiter.imagePainted.emit(self.mask_image)
        else:
            self.SignalEmiter.imagePainted.emit(self.pixmap())
        super(PainterWidget, self).mouseReleaseEvent( event)

    def clear(self):
        """ Clear the pixmap """
        self.pixmap.fill(QtCore.Qt.white)
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

class UICv_PaintNode(UIOpenCvBaseNode):
    def __init__(self, raw_node):
        super(UICv_PaintNode, self).__init__(raw_node)
        self.sizeXPin = self._rawNode.getPinByName("sizeX")
        self.sizeYPin = self._rawNode.getPinByName("sizeY")
        self.imageRefPin = self._rawNode.getPinByName("imageRef")
        self.imgPin = self._rawNode.getPinByName("img")

        self.Painter = PainterWidget(Node=self)
        self.Painter.setShapeMode(QtWidgets.QGraphicsPixmapItem.MaskShape)
        self.Painter.SignalEmiter.imagePainted.connect(self.updateimg)

        pix = QtGui.QPixmap(self.sizeXPin.getData(),self.sizeYPin.getData())
        pix.fill(QtCore.Qt.white)
        self.Painter.setPixmap(pix)       

        self.openProperties = []

    def updateimg(self,pixmap):
        self._rawNode.IMAGE = QPixmapToArray(pixmap)
        push(self.imgPin)

    def createInputWidgets(self, inputsCategory, inGroup=None, pins=True):
        
        inputsCategory.destroyed.connect(lambda x=None: self.removeItemFromViewer(inputsCategory))
        self.openProperties.append(inputsCategory)
        preIndex = inputsCategory.Layout.count()
        if pins:
            super(UICv_PaintNode, self).createInputWidgets(inputsCategory, inGroup)        

        instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
        if self.Painter.scene() == None:
            instance.viewer._scene.addItem(self.Painter)
        #self.Painter.show()

        #self.item.rotate(self.anglePin.getData())


    def removeItemFromViewer(self,obj):
        if obj in self.openProperties:
            self.openProperties.remove(obj)
        #if len(self.openProperties)==0 and self.item.scene():
        instance = self.canvasRef().pyFlowInstance.invokeDockToolByName("PyFlowOpenCv","ImageViewerTool")
        instance.viewer._scene.removeItem(self.Painter)
"""            
def on_color_clicked(self):

    color = QtWidgets.QColorDialog.getColor(QtCore.Qt.black, self)
    if color:
        self.set_color(color)

def set_color(self, color: QtGui.QColor = QtCore.Qt.black):

    # Create color icon
    pix_icon = QtGui.QPixmap(32, 32)
    pix_icon.fill(color)

    self.color_action.setIcon(QtGui.QIcon(pix_icon))
    self.painter_widget.pen.setColor(color)
    self.color_action.setText(QtGui.QColor(color).name())

"""