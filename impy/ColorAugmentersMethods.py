"""
package: Images2Dataset
class: ImageAugmenters
Author: Rodrigo Loza
Description: Common data augmentation methods for images.
"""
# from interface import Interface

# class ColorAugmentersMethods(Interface):
class ColorAugmentersMethods(object):

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

	def gaussianBlur(self, frame = None, kernelSize = None, sigma = None):
		"""
		Blur an image applying a gaussian filter with a random sigma(0, sigma_max)
		Sigma's default value is between 1 and 3.
		Args:
			frame: A tensor that contains an image.
			kernelSize: A list or tuple that contains the size of the kernel
									that will be convolved with the image.
			sigma: A float that contains the value of the gaussian filter.
		Returns:
			A tensor with a rotation of the original image.
		"""
		pass

	def averageBlur(self, frame = None, kernelSize = None):
		"""
		Convolves the image with an average filter.
		Args:
			frame: A tensor that contains an image.
			kernelSize: A tuple or list that contains the size 
									of the kernel that will be convolved with
									the image.
		Returns:
			A tensor with a blurred image.
		"""
		pass

	def medianBlur(self, frame = None, coefficient = None):
		"""
		Convolves an image with a median blur kernel.
		Args:
			frame: A tensor that contains an image.
			coefficient: An odd integer.
		Returns:
			A median blurred frame.
		"""
		pass

	def bilateralBlur(self, frame = None, d = None, sigmaColor = None, sigmaSpace = None):
		"""
		Convolves an image with a bilateral filter.
		Args:
			d: Diameter of each pixel neighborhood.
			sigmaColor: Filter color space.
			sigmaSpace: Filter the coordinate space.
		Returns:
			An image blurred by a bilateral filter.
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