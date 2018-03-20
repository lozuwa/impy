"""
package: Images2Dataset
class: DataAugmentationMethods
Author: Rodrigo Loza
Description: Common data augmentation operations 
for an image.
"""
from interface import Interface

class DataAugmentationMethods(Interface):
	
	def crop(self):
		pass

	def translation(self):
		pass

	def jitterBoxes(self):
		pass

	def horizontalFlip(self):
		pass

	def verticalFlip(self):
		pass

