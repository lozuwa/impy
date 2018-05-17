"""
package: Images2Dataset
class: ImageLocalizationDatasetPreprocessMethods
Author: Rodrigo Loza
Description: Preprocess methods for the ImageLocalization
class.
"""
# Libraries
from interface import Interface

class ImageLocalizationDatasetPreprocessMethods(Interface):
	
	def dataConsistency(self):
		"""
		Checks whether data is consistent. It starts analyzing if there is the same amount of 
		of images and annotations. Then it sees if the annotations and images are consistent 
		with each other.
		Args:
			None
		Returns:
			None
		Raises:
			- Exception: when the extension of the image is not allowed. Only jpgs and pngs are allowed.
			- Exception: When an annotation file does not have a .xml extension.
			- Exception: When the amount of annotations and images is not equal.
			- Exception: When there are images that don't have annotations.
			- Exception: When there are annotations that don't have images.
		"""
		pass

	def findEmptyOrWrongAnnotations(self, removeEmpty = None):
		"""
		Find empty or irregular annotations in the annotation files. An empty 
		annotation is an annotation that includes no objects. And a irregular 
		annotation is an annotation that has a bounding box with coordinates that
		are off the image's boundaries.
		Args:
			removeEmpty: A boolean that if True removes the annotation and image that are empty.
		Returns:
			None
		Raises:
			- Exception: when the extension of the image is not allowed. Only jpgs and pngs are allowed.
			- Exception: when an annotation file is empty.
			- Exception: when a coordinate is not valid. Either less than zero or greater than image's size.
		"""
		pass

	
