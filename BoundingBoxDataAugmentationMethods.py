"""
package: Images2Dataset
class: DataAugmentationMethods
Author: Rodrigo Loza
Description: Data augmentation methods used for bounding box 
labels.
"""
# Libraries
from interface import Interface

class BoundingBoxDataAugmentationMethods(Interface):
	
	def centerCrop(self, frameHeight = None, frameWidth = None, bndbxCoordinates = None):
		"""
		Crops the bounding box from the frame giving it a symetric 
		offset at each side so it is maintained in the center.
		Args:
			frameHeight: An int that represents the height of the frame.
			frameWidth: An int that representst he width of the frame.
			bndbxCoordinates: A tuple of ints that contains the coordinates of the
												bounding box in the frame.
			offset: An int that contains the amount of space to give at each side 
							of the bounding box.
		Returns:
			An 8-sized tuple that contains the coordinates to crop the original frame
			and the new coordinates of the bounding box inside the cropped patch.
		Example:
			Given an image and a bounding box, crop the bndbox with an offset to 
			maintain context of the object.
				---------------
				|							|
				|			---			|
				|			|	|			|
				|			---			|
				---------------
			So,
				---------------
				|		--------	|
				|		|	---		|	|
				|		|	|	|		|	|
				|		|	---		|	|
				---------------
		"""
		pass

	def cropWithTranslation(self, frameHeight = None, frameWidth = None, bndbxCoordinates = None, offset = None):
		"""
		Crops the bounding box from the frame giving it an asymetric 
		offset at each side so translation is simulated inside the context
		of the cropped patch.
		Args:
			frameHeight: An int that represents the height of the frame.
			frameWidth: An int that representst he width of the frame.
			bndbxCoordinates: A tuple of ints that contains the coordinates of the
												bounding box in the frame.
			offset: An int that contains the amount of space that will be used to limit
							the size of the cropped patch.
		Returns:
			An 8-sized tuple that contains the coordinates to crop the original frame
			and the new coordinates of the bounding box inside the cropped patch.
		Example:
			Given an image and a bounding box, crop the bndbox with an offset to 
			maintain context of the object.
				-------------------
				|							    |
				|			---					|
				|			|	|					|
				|			---					|
				|									|
				|									|
				|									|						
				-------------------
			So,
				-------------------
				|		---------			|
				|		|	---		|			|
				|		|	|	|		|			|
				|		|	---		|			|
				|		|				|			|
				|		---------			|
				|									|						
				-------------------
		"""
		pass

	def jitterBoxes(self, frame = None, quantity = None):
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