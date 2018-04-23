"""
package: Images2Dataset
class: ImageAugmenters
Author: Rodrigo Loza
Description: Common data augmentation methods for images.
"""
from interface import Interface

class GeometricAugmentersMethods(Interface):

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
