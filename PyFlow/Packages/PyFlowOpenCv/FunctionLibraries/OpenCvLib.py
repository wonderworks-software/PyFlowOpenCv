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
		img(cv2.flip(input.image,mode))	

	@staticmethod
	@IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'cv', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
	# Return a random frame of x,y
	def edgeCanny(input=('ImagePin', 0),threshold1 =('FloatPin',100.0), threshold2 =('FloatPin',200.0),
				 apertureSize =('IntPin',3),L2gradient =('BoolPin',False), img=(REF, ('ImagePin', 0))):
		"""Return a frame of the loaded image."""
		img(cv2.Canny(input.image,threshold1=threshold1,threshold2=threshold2,apertureSize=apertureSize,L2gradient=L2gradient))	

	@staticmethod
	@IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'cv', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
	# Return a random frame of x,y
	def imageBlur(input=('ImagePin', 0),x =('IntPin',5), y =('IntPin',5),
				 img=(REF, ('ImagePin', 0))):
		"""Return a frame of the loaded image."""
		img(cv2.blur(input.image,(x,y)))

	@staticmethod
	@IMPLEMENT_NODE(returns=None, meta={NodeMeta.CATEGORY: 'cv', NodeMeta.KEYWORDS: [], NodeMeta.CACHE_ENABLED: False})
	# Return a random frame of x,y
	def imageGaussianBlur(input=('ImagePin', 0),x =('IntPin',5), y =('IntPin',5),sigmaX =('IntPin',5),sigmaY =('IntPin',5),
				 img=(REF, ('ImagePin', 0))):
		"""Return a frame of the loaded image."""
		img(cv2.GaussianBlur(input.image,(x,y),sigmaX,sigmaY))	