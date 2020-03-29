from PyFlow.Core import(
    FunctionLibraryBase,
    IMPLEMENT_NODE
)
from PyFlow.Core.Common import *
import cv2
class OpenCvLib(FunctionLibraryBase):
	'''doc string for OpenCvLib'''
	def __init__(self, packageName):
		super(OpenCvLib, self).__init__(packageName)

	@staticmethod
	@IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'cv', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
	# Return a random frame of x,y
	def readImage(path=('StringPin', ""), Result=(REF, ('ImagePin', 0))):
		"""Return a frame of the loaded image."""
		image = cv2.imread(path)
		image = image[ :, :, :3 ]
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)		
		Result(image)