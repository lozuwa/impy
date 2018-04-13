"""
package: Images2Dataset
class: ImageAugmenters
Author: Rodrigo Loza
Description: Common data augmentation methods for images.
"""
from interface import Interface

class ImageAugmentersMethods(Interface):

	def addRandomBlur(self, frame = None, sigma = None):
		"""
		Blur an image applying a gaussian filter with a random sigma(0, sigma_max)
		Sigma might be between 1 and 3.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor with a rotation of the original image.
		"""
		pass

	def shiftColors(self, frame = None):
		"""
		Shifts the colors of the frame.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that has shifted the order of its colors.
		"""
		pass

	def fancyPCA(self, frame = None):
		"""
		Fancy PCA implementation.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that contains the altered image by fancy PCA.
		"""
		pass