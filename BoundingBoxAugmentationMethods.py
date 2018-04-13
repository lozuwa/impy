"""
package: Images2Dataset
class: DataAugmentationMethods
Author: Rodrigo Loza
Description: Data augmentation methods used for bounding box 
labels.
"""
# Libraries
from interface import Interface

class BoundingBoxAugmentationMethods(Interface):
	
	def jitterBoxes(self, frame = None, boundingBoxes = None, size = None, quantity = None, color = None):
		"""
		Creates random jitter boxes inside a bounding box cropped from its image.
		Args:
			frame: A tensor that contains an image.
			quantity: An int that tells how many jitter boxes to create inside 
							the frame.
		Returns:
			A tensor that contains an image altered by jitter boxes.
		"""
		pass

	def horizontalFlip(self, frame = None):
		"""
		Flip a bouding box by its horizontal axis.
		Args:
			frame: A tensor that contains a cropped bouding box from its frame.
		Returns:
			A tensor that has been flipped by its horizontal axis.
		"""
		pass

	def verticalFlip(self, frame = None):
		"""
		Flip a bouding box by its vertical axis.
		Args:
			frame: A tensor that contains a cropped bouding box from its frame.
		Returns:
			A tensor that has been flipped by its vertical axis.
		"""
		pass

	def randomRotation(self, frame = None, bndbox = None, theta = None):
		"""
		Rotate a frame clockwise by random degrees. Random degrees
		is a number that is between 20-360.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor with a rotation of the original image.
		"""
		pass
