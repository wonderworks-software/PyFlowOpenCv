from Qt import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
import os
import math
class NotImplementedException:
    pass


gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]


def toQImage(im, copy=False):
    if im is None:
        return QtGui.QImage()
        
    #if im.dtype != np.uint8:
    #	im = cv2.convertScaleAbs(im)

    if len(im.shape) == 2:
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_Indexed8)
        qim.setColorTable(gray_color_table)
        return qim.copy() if copy else qim

    elif len(im.shape) == 3:
        if im.shape[2] == 3:
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            return qim.copy() if copy else qim
        elif im.shape[2] == 4:
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
            return qim.copy() if copy else qim


    #raise NotImplementedException

MANIP_MODE_NONE = 0
MANIP_MODE_SELECT = 1
MANIP_MODE_PAN = 2
MANIP_MODE_MOVE = 3
MANIP_MODE_ZOOM = 4
MANIP_MODE_COPY = 5
class pc_ImageCanvas(QtWidgets.QGraphicsView):

    photoClicked = QtCore.Signal(QtCore.QPoint)
    def __init__(self, parent):
        super(pc_ImageCanvas, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self.fit = True
        self._photo.setShapeMode(QtWidgets.QGraphicsPixmapItem.MaskShape)
        self._scene.addItem(self._photo)

        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))

        self.setSceneRect(QtCore.QRectF(0, 0, 10, 10))

        self._manipulationMode = None
        self.mousePressPose = QtCore.QPointF(0, 0)
        self.mousePos = QtCore.QPointF(0, 0)
        self._lastMousePos = QtCore.QPointF(0, 0)

        self.centerOn(QtCore.QPointF(self.sceneRect().width() / 2, self.sceneRect().height() / 2))

        
    def currentViewScale(self):
        return self.transform().m22()

    def hasPhoto(self):
        return not self._empty

    def frameRect(self, rect):
        if rect is None:
            return
        windowRect = self.mapToScene(self.rect()).boundingRect()

        # pan to center of window
        delta = windowRect.center() - rect.center()
        delta *= self.currentViewScale()
        self.pan(delta)

        # zoom to fit content
        ws = windowRect.size()
        rect += QtCore.QMargins(40, 40, 40, 40)
        rs = rect.size()
        widthRef = ws.width()
        heightRef = ws.height()
        sx = widthRef / rect.width()
        sy = heightRef / rect.height()
        scale = sx if sy > sx else sy
        self.zoom(scale)

        return scale

    def frameItems(self, items):
        rect = QtCore.QRect()
        for i in items:
            rect |= i.sceneBoundingRect().toRect()
        self.frameRect(rect)

    def fitInView(self, scale=True, factor=1):
        if self.hasPhoto():
            viewrect = self.viewport().rect()
            if scale and viewrect.width() > 0 and viewrect.height() > 0:
                self.frameItems([self._photo])
            else:
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(factor / unity.width(), factor / unity.height())

    def setNumpyArray(self, image):
        if image.__class__.__name__ == "UMat":
            image = cv2.UMat.get(image)
        image = toQImage(image)  # self.createQimagefromNumpyArray(image)
        pixmap = QtGui.QPixmap.fromImage(image, QtCore.Qt.ThresholdAlphaDither)
        self.setPhoto(pixmap)

    def setPhoto(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
            if self.fit:
                self.fitInView(True)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())

    def pan(self, delta):
        rect = self.sceneRect()
        scale = self.currentViewScale()
        x = -delta.x() / scale
        y = -delta.y() / scale
        rect.translate(x, y)
        self.setSceneRect(rect)
        self.update()

    def wheelEvent(self, event):
        self.fit = False
        (xfo, invRes) = self.transform().inverted()
        topLeft = xfo.map(self.rect().topLeft())
        bottomRight = xfo.map(self.rect().bottomRight())
        center = (topLeft + bottomRight) * 0.5
        zoomFactor = 1.0 + event.delta() * 0.0005

        self.zoom(zoomFactor)

    def zoom(self, scale_factor):
        self.factor = self.transform().m22()
        futureScale = self.factor * scale_factor
        if futureScale <= 0.01:
            scale_factor = (0.01) / self.factor
        if futureScale >= 100:
            scale_factor = (100 - 0.1) / self.factor
        self.scale(scale_factor, scale_factor)


    def mousePressEvent(self, event):

        modifiers = event.modifiers()
        self.mousePressPose = event.pos()
        if event.button() == QtCore.Qt.LeftButton and modifiers in [QtCore.Qt.NoModifier,QtCore.Qt.ShiftModifier,QtCore.Qt.ControlModifier]:
            self._manipulationMode = MANIP_MODE_SELECT
            self.viewport().setCursor(QtCore.Qt.ArrowCursor)
            super(pc_ImageCanvas, self).mousePressEvent(event)

        LeftPaning = event.button() == QtCore.Qt.LeftButton and modifiers == QtCore.Qt.AltModifier
        if event.button() == QtCore.Qt.MiddleButton or LeftPaning:
            self.viewport().setCursor(QtCore.Qt.OpenHandCursor)
            self._manipulationMode = MANIP_MODE_PAN
            self._lastPanPoint = self.mapToScene(event.pos())                
        elif event.button() == QtCore.Qt.RightButton:
            self.viewport().setCursor(QtCore.Qt.SizeHorCursor)
            self._manipulationMode = MANIP_MODE_ZOOM
            self._lastMousePos = event.pos()
            self._lastTransform = QtGui.QTransform(self.transform())
            self._lastSceneRect = self.sceneRect()
            self._lastSceneCenter = self._lastSceneRect.center()
            self._lastScenePos = self.mapToScene(event.pos())
            self._lastOffsetFromSceneCenter = self._lastScenePos - self._lastSceneCenter       
        #super(pc_ImageCanvas, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mousePos = event.pos()
        modifiers = event.modifiers()
        if self._manipulationMode == MANIP_MODE_SELECT :
            super(pc_ImageCanvas, self).mouseMoveEvent(event)
        elif self._manipulationMode == MANIP_MODE_PAN:
            delta = self.mapToScene(event.pos()) - self._lastPanPoint
            rect = self.sceneRect()
            rect.translate(-delta.x(), -delta.y())
            self.setSceneRect(rect)
            self._lastPanPoint = self.mapToScene(event.pos())

        elif self._manipulationMode == MANIP_MODE_ZOOM:

           # How much
            delta = event.pos() - self._lastMousePos
            #self._lastMousePos = event.pos()
            zoomFactor = 1.0
            if delta.x() > 0:
                zoomFactor = 1.0 + delta.x() / 100.0
            else:
                zoomFactor = 1.0 / (1.0 + abs(delta.x()) / 100.0)

            # Limit zoom to 3x
            if self._lastTransform.m22() * zoomFactor >= 2.0:
                return

            # Reset to when we mouse pressed
            self.setSceneRect(self._lastSceneRect)
            self.setTransform(self._lastTransform)

            # Center scene around mouse down
            rect = self.sceneRect()
            rect.translate(self._lastOffsetFromSceneCenter)
            self.setSceneRect(rect)

            # Zoom in (QGraphicsView auto-centers!)
            self.scale(zoomFactor, zoomFactor)

            newSceneCenter = self.sceneRect().center()
            newScenePos = self.mapToScene(self._lastMousePos)
            newOffsetFromSceneCenter = newScenePos - newSceneCenter

            # Put mouse down back where is was on screen
            rect = self.sceneRect()
            rect.translate(-1 * newOffsetFromSceneCenter)
            self.setSceneRect(rect)

            # Call udpate to redraw background
            self.update()

        else:
            super(pc_ImageCanvas, self).mouseMoveEvent(event)    
        #if self.itemAt(event.pos()) == self._photo :
        p = self._photo.mapFromScene(self.mapToScene(event.pos()))
        pixel_pos = p.toPoint()
        #print pixel_pos
    def mouseReleaseEvent(self, event):
        super(pc_ImageCanvas, self).mouseReleaseEvent(event)
        self.mouseReleasePos = event.pos()
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)

        self._manipulationMode = MANIP_MODE_NONE        

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_F:
            self.fit = True
            self.fitInView(True)
        if key in range(49, 58):
            self.fit = False
            self.fitInView(False, max(0, key - 48))

    def clear2(self):
        self.setPhoto(None)
# self.clear()



if __name__ == '__main__':
    import sys
    from Qt import QtSvg
    app = QtWidgets.QApplication(sys.argv)
    window = pc_ImageCanvas(None)
    window.setPhoto(QtGui.QPixmap(r"C:\Users\pedro\OneDrive\pcTools_v5\PyFlowPackages\PyFlowOpenCv\images\shapes_and_colors.png"))
 

    window.setGeometry(500, 300, 800, 600)
    window.show()
 
    sys.exit(app.exec_())
