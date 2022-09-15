PACKAGE_NAME = 'PyFlowOpenCv'

from collections import OrderedDict
from PyFlow.UI.UIInterfaces import IPackage

# Pins
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import ImagePin
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import VideoPin
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import GraphElementPin
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import KeyPointsPin
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import BackgroundSubtractorPin
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import DescriptorPin
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import FeatureMatchPin
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import NumpyDataPin

# Function based nodes
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.OpenCvLib import OpenCvLib
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.OpticalFlowLib import OpticalFlowLib
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.OpenCvLib import LK_optical_flow_Lib
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.OpenCvLib import Dense_optical_flow_Lib
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.ImageFilteringLib import ImageFilteringLib
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.GeometricImageTransformationsLib import GeometricImageTransformationsLib

# Class based nodes
from PyFlow.Packages.PyFlowOpenCv.Nodes.ViewerNode import ViewerNode
from PyFlow.Packages.PyFlowOpenCv.Nodes.PaintNode import PaintNode

# Tools
from PyFlow.Packages.PyFlowOpenCv.Tools.ImageViewerTool import ImageViewerTool

# Factories
from PyFlow.Packages.PyFlowOpenCv.Factories.PinInputWidgetFactory import getInputWidget
from PyFlow.Packages.PyFlowOpenCv.Factories.UINodeFactory import createUINode

_FOO_LIBS = {OpenCvLib.__name__: OpenCvLib(PACKAGE_NAME),
			LK_optical_flow_Lib.__name__: LK_optical_flow_Lib(PACKAGE_NAME),
			Dense_optical_flow_Lib.__name__:Dense_optical_flow_Lib(PACKAGE_NAME),
			ImageFilteringLib.__name__: ImageFilteringLib(PACKAGE_NAME),
			GeometricImageTransformationsLib.__name__: GeometricImageTransformationsLib(PACKAGE_NAME),
			OpticalFlowLib.__name__: OpticalFlowLib(PACKAGE_NAME),
			}
_NODES = {}
_PINS = {}
_TOOLS = OrderedDict()
_PREFS_WIDGETS = OrderedDict()
_EXPORTERS = OrderedDict()


_NODES[ViewerNode.__name__] = ViewerNode
_NODES[PaintNode.__name__] = PaintNode

_PINS[ImagePin.__name__] = ImagePin
_PINS[VideoPin.__name__] = VideoPin
_PINS[GraphElementPin.__name__] = GraphElementPin
_PINS[KeyPointsPin.__name__] = KeyPointsPin
_PINS[BackgroundSubtractorPin.__name__] = BackgroundSubtractorPin
_PINS[DescriptorPin.__name__] = DescriptorPin
_PINS[FeatureMatchPin.__name__] = FeatureMatchPin
_PINS[NumpyDataPin.__name__] = NumpyDataPin

_TOOLS[ImageViewerTool.__name__] = ImageViewerTool


class PyFlowOpenCv(IPackage):
	def __init__(self):
		super(PyFlowOpenCv, self).__init__()

	@staticmethod
	def GetExporters():
		return _EXPORTERS

	@staticmethod
	def GetFunctionLibraries():
		return _FOO_LIBS

	@staticmethod
	def GetNodeClasses():
		return _NODES

	@staticmethod
	def GetPinClasses():
		return _PINS

	@staticmethod
	def GetToolClasses():
		return _TOOLS

	#@staticmethod
	#def UIPinsFactory():
	#	return createUIPin

	@staticmethod
	def UINodesFactory():
		return createUINode

	@staticmethod
	def PinsInputWidgetFactory():
		return getInputWidget

