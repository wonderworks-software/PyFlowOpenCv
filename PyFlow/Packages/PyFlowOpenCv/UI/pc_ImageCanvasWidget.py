from Qt import QtCore, QtGui, QtWidgets
import numpy as np


class NotImplementedException:
    pass


gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]


def toQImage(im, copy=False):
    if im is None:
        return QtGui.QImage()

    if im.dtype == np.uint8:
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

    raise NotImplementedException


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

        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True, factor=1):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                viewrect = self.viewport().rect()
                if scale and viewrect.width() > 0 and viewrect.height() > 0:
                    scenerect = self.transform().mapRect(rect)
                    factor = min(viewrect.width() / scenerect.width(),
                                 viewrect.height() / scenerect.height())
                    self.scale(factor, factor)
                else:
                    unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                    self.scale(factor / unity.width(), factor / unity.height())
            self._zoom = 0

    def createQimagefromNumpyArray(self, img):
        i = QtGui.QImage(img, img.shape[1], img.shape[0], img.shape[1] * img.shape[2], QtGui.QImage.Format_RGB888)
        return i

    def setNumpyArray(self, image):
        image = toQImage(image)  # self.createQimagefromNumpyArray(image)
        pixmap = QtGui.QPixmap.fromImage(image, QtCore.Qt.ThresholdAlphaDither)
        self.setPhoto(pixmap)

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
            if self.fit:
                self.fitInView(True)
        # else:
        #	self.fitInView(False)
        else:
            self._empty = True
            self.setDragMode(QtGui.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())

    def wheelEvent(self, event):
        if self.hasPhoto():
            self.fit = False
            if event.delta() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        super(pc_ImageCanvas, self).mouseMoveEvent(event)
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        super(pc_ImageCanvas, self).mouseReleaseEvent(event)
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, event):
        super(pc_ImageCanvas, self).mousePressEvent(event)
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)

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

    app = QtWidgets.QApplication(sys.argv)
    window = pc_ImageCanvas(None)
    window.setPhoto(QtGui.QPixmap(r"..\logo.png"))
    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec_())
