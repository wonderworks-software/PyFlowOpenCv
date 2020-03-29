from PyFlow.UI.Canvas.UINodeBase import UINodeBase

from PyFlow.Packages.PyFlowOpenCv.UI.UIOpenCvBaseNode import UIOpenCvBaseNode
def createUINode(raw_instance):
    return UIOpenCvBaseNode(raw_instance)
