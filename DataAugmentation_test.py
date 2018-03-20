"""
package: Images2Dataset
class: DataAugmentation
Author: Rodrigo Loza
Description: Test class for data augmentation.
"""
# Libraries
import unittest
import numpy as np 
import cv2

class DataAugmentation_test(unittest.TestCase):
	
	def setUp(self):
		dummy_image = np.random.rand(300,300)
		self.data_augmentation = dataAugmentation(frame = dummy_image)

	def test_crop(self):
		image_result = self.data_augmentation.crop()

	def test_translation(self):
		image_result = self.data_augmentation.translation()

	def test_jitterBoxes(self):
		image_result = self.data_augmentation.jitterBoxes()

	def test_horizontalFlip(self):
		image_result = self.data_augmentation.horizontalFlip()

	def test_verticalFlip(self):
		image_result = self.data_augmentation.verticalFlip()

if __name__ == "__main__":
	unittest.main()
