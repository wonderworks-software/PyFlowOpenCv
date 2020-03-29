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
	def readImage(path=('StringPin', ""), img=(REF, ('ImagePin', 0))):
		"""Return a frame of the loaded image."""
		image = cv2.imread(path)
		image = image[ :, :, :3 ]
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)		
		img(image)

	@staticmethod
	@IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'cv', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
	# Return a random frame of x,y
	def flipImage(input=('ImagePin', 0),mode=('IntPin',0), img=(REF, ('ImagePin', 0))):
		"""Return a frame of the loaded image."""
		image = input.image
		img(cv2.flip(image,mode))		