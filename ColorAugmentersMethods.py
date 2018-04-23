"""
package: Images2Dataset
class: ImageAugmenters
Author: Rodrigo Loza
Description: Common data augmentation methods for images.
"""
from interface import Interface

class ColorAugmentersMethods(Interface):

	def invertColor(self, frame = None, CSpace = None):
		"""
		Inverts the color of an image.
		Args:
			frame: A tensor that contains an image.
			CSpace: A 3-sized tuple that contains booleans (B, G, R).
							If a boolean is set to true, then we invert that channel.
							If the 3 booleans are false, then we invert all the image.
		Returns:
			A tensor that has its color inverted.
		"""
		pass

	def histogramEqualization(self, frame = None, equalizationType = None):
		"""
		Args:
			frame: A tensor that contains an image.
			equalizationType: An int that defines what type of histogram
						equalization algorithm to use.
		Returns:
			A frame whose channels have been equalized.
		"""
		pass

	def changeBrightness(self, frame = None, coefficient = None):
		"""
		Change the brightness of a frame.
		Args:
			frame: A tensor that contains an image.
			coefficient: A float that changes the brightness of the image.
									Default is a random number in the range of 2.
		Returns:
			A tensor with its brightness property changed.
		"""
		pass

	def sharpening(self, frame = None, weight = None):
		"""
		Sharpens an image.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A sharpened tensor.
		"""
		pass
	
	def addGaussianNoise(self, frame = None, coefficient = None):
		"""
		Add gaussian noise to a tensor.
		Args:
			frame: A tensor that contains an image.
			coefficient: A float that contains the amount of noise to add
										to a frame.
		Returns:
			An altered frame that has gaussian noise.
		"""
		pass

	def gaussianBlur(self, frame = None, sigma = None):
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