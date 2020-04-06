from PyFlow.UI.Canvas.UINodeBase import UINodeBase

from PyFlow.Packages.PyFlowOpenCv.UI.UIOpenCvBaseNode import UIOpenCvBaseNode
from PyFlow.Packages.PyFlowOpenCv.UI.UICv_TransformNode import UICv_TransformNode

def createUINode(raw_instance):
	if raw_instance.__class__.__name__ == "cv_Transform":
		return UICv_TransformNode(raw_instance)
	return UIOpenCvBaseNode(raw_instance)
