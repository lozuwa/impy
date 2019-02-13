"""
package: Images2Dataset
class: DataAugmentationMethods
Author: Rodrigo Loza
Description: Data augmentation methods used for bounding box 
labels.
"""
# Libraries
# from interface import Interface

# class BoundingBoxAugmentersMethods(Interface):
class BoundingBoxAugmentersMethods(object):
	
	def scale(self, frame = None, boundingBoxes = None, size = None, zoom = None, interpolationMethod = None):
		"""
		Scales an image with its bounding boxes to another size while maintaing the 
		coordinates of the bounding boxes.
		Args:
			frame: A tensor that contains an image.
			boundingBoxes: A list of lists that contains the coordinates of the bounding
											boxes that are part of the image.
			size: A tuple that contains the resizing values.
			zoom: A boolean that defines if scaling will be executed as zoom.
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

	def crop(self, boundingBoxes = None, size = None):
		"""
		Crop a list of bounding boxes.
		Args:
			boundingBoxes: A list of lists that contains the coordinates of bounding 
										boxes.
			size: A tuple that contains the size of the crops to be performed.
		Returns:
			A list of lists with the updated coordinates of the bounding boxes after 
			being cropped.
		"""
		pass

	def pad(self, frameHeight = None, frameWidth = None, boundingBoxes = None, size = None):
		"""
		Includes pixels from outside the bounding box as padding.
		Args:
			frameHeight: An int that contains the height of the frame.
			frameWidth: An int that contains the width of the frame.
			boundingBoxes: A list of lists that contains coordinates of bounding boxes.
			size: A tuple that contains the size of pixels to pad the image with.
		Returns:
			A list of lists that contains the coordinates of the bounding
			boxes padded with exterior pixels of the parent image.
		"""
		pass

	def jitterBoxes(self, frame = None, boundingBoxes = None, size = None, quantity = None, color = None):
		"""
		Draws random jitter boxes in the bounding boxes.
		Args:
			frame: A tensor that contains an image.
			boundingBoxes: A list of lists that contains the coordinates of the boudning
										boxes that belong to the frame.
			size: A tuple that contains the size of the jitter boxes to draw.
			quantity: An int that tells how many jitter boxes to draw inside 
							each bounding box.
			color: A 3-sized tuple that contains some RGB color. If default it is black.
		Returns:
			A tensor that contains an image altered by jitter boxes.
		"""
		pass

	def horizontalFlip(self, frame = None, boundingBoxes = None):
		"""
		Flip a bouding box by its horizontal axis.
		Args:
			frame: A tensor that contains an image with its bounding boxes.
			boundingBoxes: A list of lists that contains the coordinates of the bounding
											boxes that belong to the tensor.
		Returns:
			A tensor whose bounding boxes have been flipped by its horizontal axis.
		"""
		pass

	def verticalFlip(self, frame = None, boundingBoxes = None):
		"""
		Flip a bouding box by its vertical axis.
		Args:
			frame: A tensor that contains a cropped bouding box from its frame.
			boundingBoxes: A list of lists that contains the coordinates of the bounding
											boxes that belong to the tensor.
		Returns:
			A tensor whose bounding boxes have been flipped by its vertical axis.
		"""
		pass

	def rotation(self, frame = None, boundingBoxes = None, theta = None):
		"""
		Rotate the bounding boxes of a frame clockwise by n degrees. The degrees are
		in the range of 20-360.
		Args:
			frame: A tensor that contains an image.
			bndbox: A list of lists that contains the coordinates of the bounding boxes
						 in the image.
			theta: An int that contains the amount of degrees to move.
							Default is random.
		Returns:
			A tensor that contains the rotated image and a tuple
			that contains the rotated coordinates of the bounding box.
		"""
		pass

	def dropout(self, frame = None, boundingBoxes = None, size = None, threshold = None, color = None):
		"""
		Set pixels inside a bounding box to zero depending on probability p 
		extracted from a normal distribution with zero mean and one standard deviation.
		Args:
			frame: A tensor that contains an image.
			boundingBoxes: A list of lists that contains the coordinates of the bounding
											boxes that belong to the frame.
			size: A tuple that contains the size of the regions that will be randomly
						set to zero according to a dropout scenario.
			threshold: A float that contains the probability threshold for the dropout
									scenario.
		Returns:
			A tensor with the altered pixels.
		"""
		pass