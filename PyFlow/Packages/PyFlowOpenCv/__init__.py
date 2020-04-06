PACKAGE_NAME = 'PyFlowOpenCv'

from collections import OrderedDict
from PyFlow.UI.UIInterfaces import IPackage

# Pins
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import ImagePin
from PyFlow.Packages.PyFlowOpenCv.Pins.ImagePin import VideoPin

# Function based nodes
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.OpenCvLib import OpenCvLib
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.ImageFilteringLib import ImageFilteringLib
from PyFlow.Packages.PyFlowOpenCv.FunctionLibraries.GeometricImageTransformationsLib import GeometricImageTransformationsLib

# Class based nodes
from PyFlow.Packages.PyFlowOpenCv.Nodes.ViewerNode import ViewerNode

# Tools
from PyFlow.Packages.PyFlowOpenCv.Tools.ImageViewerTool import ImageViewerTool

# Factories
from PyFlow.Packages.PyFlowOpenCv.Factories.PinInputWidgetFactory import getInputWidget
from PyFlow.Packages.PyFlowOpenCv.Factories.UINodeFactory import createUINode

_FOO_LIBS = {OpenCvLib.__name__: OpenCvLib(PACKAGE_NAME),
			ImageFilteringLib.__name__: ImageFilteringLib(PACKAGE_NAME),
			GeometricImageTransformationsLib.__name__: GeometricImageTransformationsLib(PACKAGE_NAME),
			}
_NODES = {}
_PINS = {}
_TOOLS = OrderedDict()
_PREFS_WIDGETS = OrderedDict()
_EXPORTERS = OrderedDict()


_NODES[ViewerNode.__name__] = ViewerNode

_PINS[ImagePin.__name__] = ImagePin
_PINS[VideoPin.__name__] = VideoPin

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

