"""
package: Images2Dataset
class: ImageAugmenters
Author: Rodrigo Loza
Description: Common data augmentation methods for images.
"""
from interface import Interface

class ImageAugmentersMethods(Interface):

	def scale(self, frame = None, size = None, interpolationMethod = None):
		"""
		Scales an image with its bounding boxes to another size while maintaing the 
		coordinates of the bounding boxes.
		Args:
			frame: A tensor that contains an image.
			boundingBoxes: A list of lists that contains the coordinates of the bounding
											boxes that are part of the image.
			resizeSize: A tuple that contains the resizing values.
			interpolationMethod: Set the type of interpolation method. 
														(INTER_NEAREST -> 0,
														INTER_LINEAR -> 1, 
														INTER_CUBIC -> 2, 
														INTER_LANCZOS4 -> 4)
		Returns:
			An image that has been scaled and a list of lists that contains the new 
			coordinates of the bounding boxes.
		"""
		pass

	def translate(self, frame = None, offset = None):
		"""
		Given an image and its bounding boxes, this method translates the bounding boxes
		to create an alteration of the image.
		Args:
			frame: A tensor that contains an image.
			offset: A tuple that contains the amoung of space to move on each axis.
							(widthXheight)
		Returns:
			A translated tensor by offset.
		"""
		pass

	def jitterBoxes(self, frame = None, size = None, quantity = None, color = None):
		"""
		Draws random jitter boxes in the bounding boxes.
		Args:
			frame: A tensor that contains an image.
			size: A tuple that contains the size of the jitter boxes to draw.
			quantity: An int that tells how many jitter boxes to create inside 
							the frame.
			color: A 3-sized tuple that contains the RGB code for a color. Default
							is black (0,0,0)
		Returns:
			A tensor that contains an image altered by jitter boxes.
		"""
		pass

	def horizontalFlip(self, frame = None):
		"""
		Flip a frame by its horizontal axis.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that has been flipped by its horizontal axis.
		"""
		pass

	def verticalFlip(self, frame = None):
		"""
		Flip a bouding box by its vertical axis.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that has been flipped by its vertical axis.
		"""
		pass

	def rotation(self, frame = None, bndbox = None, theta = None):
		"""
		Rotate a frame clockwise by random degrees. Random degrees
		is a number that is between 20-360.
		Args:
			frame: A tensor that contains an image.
			bndbox: A tuple that contains the ix, iy, x, y coordinates 
							of the bounding box in the image.
			theta: An int that contains the amount of degrees to move.
							Default is random.
		Returns:
			A tensor that contains an image.
		"""
		pass

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